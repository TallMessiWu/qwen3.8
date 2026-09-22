#!/usr/bin/env python3
"""Single-operator test for the DELIVERED QFA (cann_ops_transformer), PA_NZ layout.

Why this exists next to test_qfa_op.py: that script drives
``torch.ops._C_ascend.npu_quant_flash_attn{,_metadata}`` -- the copy vendored
into junlin-qfa's csrc -- against a CPU golden, in PA_BBND/PA_BNBD, and its one
graph case runs the metadata op OUTSIDE the graph into a fixed buffer ("D4
mode"). The junlin-c8-mxfp* branches call something else entirely:
``cann_ops_transformer.ops.quant_flash_attn{,_metadata}``, in PA_NZ, with the
metadata op INSIDE the captured region. Nothing covered that combination, so a
CANN/ops-transformer package swap could only be discovered by starting a server.

Scope: interface contract and capture behaviour, NOT numeric accuracy. There is
no CPU golden here on purpose -- porting one would pin this script to a single
operator version, which is exactly what it is supposed to survive. Accuracy
lives in test_qfa_op.py; what this script asserts is that the delivered
operator still honours the four assumptions attention_v1.py is built on:

  SIG            every keyword _get_qfa_metadata()/_run_qfa() passes still
                 exists on the delivered wrappers (hard gate -- the rest of
                 the run is meaningless if the signature moved)
  DECODE         eager Q_S=1, mask_mode=NO_MASK, q_descale N2TGD
  PREFILL        eager varlen, mask_mode=CAUSAL, q_descale TND
  PLAN-SIZE      does the metadata plan's size depend on batch / max_seqlen_q?
                 Prints STATIC or DYNAMIC. attention_v1.py caches one plan per
                 step keyed on the non-tensor inputs only, and under capture
                 the plan tensor is allocated once -- a size that moves with
                 the tensor CONTENTS breaks both.
  PLAN-RO        one plan fed to three consecutive main-op calls: outputs must
                 be bit-exact and the plan's bytes unchanged. attention_v1.py
                 shares one plan across all 23 full-attention layers of a step
                 because the operator declares ``metadata`` as a read-only
                 Input. If the new delivery writes state back into it, layer 1
                 is right and layers 2..N are quietly wrong.
  VDESC          the (1,1,1,1,1,2) v_descale placeholder
                 (_qfa_v_descale_placeholder: quant_mode=1 refuses a null
                 v_descale, but nothing reads it under PA_NZ) must still be
                 accepted AND give bit-exact results against a full-size
                 v_descale.
  GRAPH-DECODE   npugraph_ex capture with the metadata op inside the compiled
                 region, inputs swapped in place, replay vs eager bit-exact
  GRAPH-PREFILL  same for CAUSAL + TND q_descale -- the case no existing
                 script covers, and the shape PD-disaggregated P instances run

Limits worth knowing before trusting a GREEN:
  - The graph cases use torch.compile(backend="npugraph_ex") directly. The
    engine reaches npugraph_ex through vLLM's compilation config, so this
    reproduces the operator's capture behaviour, not the engine's plumbing.
  - Length tensors here are built the way _qfa_step_lengths builds them
    (persistent int32 buffers, clamp + cummax on device) but the buffers are
    refreshed by this script, not by _prepare_inputs.
  - No CPU golden: a wrong-but-stable operator passes. Pair with
    test_qfa_op.py when the numerics themselves are in question.

Usage (inside the serving container, on one NPU):
    python3 scripts/bench/test_qfa_cann_ops.py
    python3 scripts/bench/test_qfa_cann_ops.py --model 27b
    python3 scripts/bench/test_qfa_cann_ops.py --cases SIG,PLAN-SIZE,PLAN-RO
    python3 scripts/bench/test_qfa_cann_ops.py --worktree /home/hajimi/qwen3.8/vllm-ascend/junlin-c8-mxfp-16614

Prints [GREEN]/[RED]/[SKIP] per case, exits non-zero on any RED.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import sys
import traceback
from pathlib import Path

import torch

# QFA constants, mirrored from attention_v1.py.
QUANT_MODE_MXFP8 = 1
MASK_MODE_NO_MASK = 0
MASK_MODE_CAUSAL = 3
LAYOUT_TND = "TND"
LAYOUT_N2TGD = "N2TGD"
LAYOUT_PA_NZ = "PA_NZ"
FP8_DTYPE = torch.float8_e4m3fn
E8M0_DTYPE = torch.float8_e8m0fnu

# Keyword sets attention_v1.py actually passes. Keep in sync with
# _get_qfa_metadata() and _run_qfa(); qfa_op_contract.py carries the same lists
# for the no-NPU introspection path.
METADATA_KWARGS = [
    "cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv", "v_descale",
    "max_seqlen_q", "max_seqlen_kv", "mask_mode", "win_left", "win_right",
    "layout_q", "layout_q_descale", "layout_kv", "layout_out",
]
MAIN_KWARGS = [
    "block_table", "p_scale", "cu_seqlens_q", "cu_seqlens_kv", "seqused_q",
    "seqused_kv", "sinks", "attn_mask", "metadata", "softmax_scale",
    "mask_mode", "win_left", "win_right", "max_seqlen_q", "max_seqlen_kv",
    "layout_q", "layout_q_descale", "layout_kv", "layout_out",
    "return_softmax_lse",
]

# Per-rank head shapes. C8_MXFP pages are 512 tokens
# (AscendC8MXFPAttentionBackend.get_supported_kernel_block_sizes).
MODELS = {
    "2.4t": dict(num_heads=16, num_kv_heads=2, head_dim=256, block_size=512),
    "27b": dict(num_heads=24, num_kv_heads=4, head_dim=256, block_size=512),
}

DEFAULT_WORKTREES = (
    "/home/hajimi/qwen3.8/vllm-ascend/junlin-c8-mxfp-16614",
    "/home/hajimi/qwen3.8/vllm-ascend/junlin-c8-mxfp-16278",
    "/home/hajimi/qwen3.8/vllm-ascend/junlin-c8-mxfp",
)


# --------------------------------------------------------------------------
# Operator + layout-helper resolution
# --------------------------------------------------------------------------
def get_qfa_ops():
    """Resolve the delivered dual operators, exactly as _get_qfa_ops() does."""
    from cann_ops_transformer.ops import quant_flash_attn as main_op
    from cann_ops_transformer.ops import quant_flash_attn_metadata as metadata_op
    return main_op, metadata_op


def load_mxfp_helpers(worktree: str | None):
    """Import the engine's own PA_NZ shape/scatter helpers.

    Same source as the engine, so a layout change lands here too instead of
    silently drifting (test_qfa_vs_fia.py copies its helpers and says so).
    Tries the installed package first, then loads the module by path -- by
    path rather than by package to skip vllm_ascend.device.__init__, which
    pulls in the hardware profile. mxfp_kv_cache itself imports only torch
    and torch_npu.
    """
    try:
        import vllm_ascend.device.mxfp_kv_cache as mod  # type: ignore[import-not-found]
        return mod, "installed package"
    except Exception:  # noqa: BLE001, S110 -- any failure just means fall back to the path load
        pass
    candidates = [worktree] if worktree else list(DEFAULT_WORKTREES)
    for root in candidates:
        if not root:
            continue
        path = Path(root) / "vllm_ascend" / "device" / "mxfp_kv_cache.py"
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("_qfa_mxfp_kv_cache", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, str(path)
    raise RuntimeError(
        "cannot locate vllm_ascend/device/mxfp_kv_cache.py -- pass --worktree "
        f"(tried the installed package and {list(candidates)})"
    )


# --------------------------------------------------------------------------
# Data construction (PA_NZ, built the way the engine builds it)
# --------------------------------------------------------------------------
def nz_5d_view(cache: torch.Tensor, num_kv_heads: int, head_dim: int, block_size: int):
    """(num_blocks, block_size, num_kv_heads, dim) -> PA_NZ 5-D, as _nz_5d_view."""
    return cache.view(-1, num_kv_heads, head_dim // 32, block_size, 32)


def q_scale_tnd_to_n2tgd(scale_tnd: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """(T, Nq, Dg, 2) -> (Nkv, T, G, Dg, 2), mirroring _qfa_query_scale_for_layout.

    Permutes the uint8 byte view: transpose/contiguous on float8 either errors
    or falls back to AICPU.
    """
    src = scale_tnd.view(torch.uint8) if scale_tnd.dtype != torch.uint8 else scale_tnd
    t, n, dg, two = src.shape
    g = n // num_kv_heads
    return src.reshape(t, num_kv_heads, g, dg, two).permute(1, 0, 2, 3, 4).contiguous()


def causal_mask_npu() -> torch.Tensor:
    """int8 2048x2048, 1 = masked future (what the shipped op tests use)."""
    return torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()


def as_e8m0(t: torch.Tensor) -> torch.Tensor:
    """Bitcast to E8M0. torch_npu.float8_e8m0fnu is the integer dtype ID (293)
    on this build, not a torch.dtype -- tensor.view() would read it as a shape."""
    return t if t.dtype == E8M0_DTYPE else t.view(E8M0_DTYPE)


class Batch:
    """One step's worth of device state: caches, block table, length buffers.

    The caches are allocated once at full capacity and refilled in place, so a
    captured graph can be replayed against a second batch without any address
    changing -- the engine's contract.
    """

    def __init__(self, shape: dict, max_blocks_per_req: int, max_batch: int, seed: int):
        import torch_npu  # noqa: F401  (registers the npu device + ops)

        self.nq = shape["num_heads"]
        self.nkv = shape["num_kv_heads"]
        self.d = shape["head_dim"]
        self.block_size = shape["block_size"]
        self.max_blocks_per_req = max_blocks_per_req
        self.max_batch = max_batch
        self.num_blocks = max_batch * max_blocks_per_req
        self.softmax_scale = self.d ** -0.5
        self.mxfp = None  # set by allocate()

        torch.manual_seed(seed)

    def allocate(self, mxfp):
        self.mxfp = mxfp
        nb, bs, nkv, d = self.num_blocks, self.block_size, self.nkv, self.d
        # K/V keep the natural 4-D storage; the NZ view is taken at the call.
        self.k_cache = torch.zeros(nb, bs, nkv, d, dtype=torch.int8).npu()
        self.v_cache = torch.zeros(nb, bs, nkv, d, dtype=torch.int8).npu()
        self.k_scale_cache = torch.zeros(
            *mxfp.mxfp_k_scale_cache_shape(nb, bs, nkv, d), dtype=torch.uint8).npu()
        self.v_scale_cache = torch.zeros(
            *mxfp.mxfp_v_scale_cache_shape(nb, bs, nkv, d), dtype=torch.uint8).npu()
        # V's scale is static per channel and filled once, before any request
        # or capture -- exactly like the engine (fill_mxfp_v_scale_cache).
        v_scale_bytes = torch.randint(120, 132, (nkv * d,), dtype=torch.uint8)
        mxfp.fill_mxfp_v_scale_cache(v_scale_bytes.npu(), self.v_scale_cache)
        # npu_quantize requires the scale's dtype to match x's, and x here is
        # bf16 -- passing fp32 fails with EZ1001 "dtype of input x:DT_BFLOAT16
        # is not compatible with scale:DT_FLOAT". The engine builds this the
        # same way: (1 / exp2(e)).to(model dtype), i.e. bf16
        # (mxfp_c8.py process_weights_after_loading).
        self.v_recip = (1.0 / torch.exp2(v_scale_bytes.float() - 127.0)).bfloat16().npu()

        self.block_table = torch.zeros(self.max_batch, self.max_blocks_per_req,
                                       dtype=torch.int32).npu()
        # Persistent int32 length buffers, refreshed in place per step -- the
        # runner's query_start_loc_gpu / seq_lens_gpu stand-ins.
        self.qsl_buf = torch.zeros(self.max_batch + 1, dtype=torch.int32).npu()
        self.seq_lens_buf = torch.zeros(self.max_batch, dtype=torch.int32).npu()
        self.mask = causal_mask_npu()

    def load(self, q_lens: list[int], kv_lens: list[int]):
        """Fill this step's K/V, scales, block table and length buffers in place.

        Returns (num_tokens, q_fp8, q_scale_tnd) -- the query side, freshly
        quantized with the same npu_dynamic_mx_quant the engine uses.
        """
        import torch_npu

        b = len(q_lens)
        assert b <= self.max_batch and len(kv_lens) == b
        assert all(kv <= self.max_blocks_per_req * self.block_size for kv in kv_lens)
        num_tokens = sum(q_lens)

        # Block table: request i owns a contiguous run of its own blocks, so no
        # two requests alias (block 0 stays the null block, as in vLLM).
        table = torch.zeros(self.max_batch, self.max_blocks_per_req, dtype=torch.int32)
        for i in range(b):
            base = i * self.max_blocks_per_req
            table[i] = torch.arange(base, base + self.max_blocks_per_req, dtype=torch.int32)
        self.block_table.copy_(table.npu())

        # Length buffers, with the tail left as the runner leaves it: -1 in
        # query_start_loc (FIA padding convention), 0 in seq_lens. The clamp +
        # cummax in step_lengths() is what turns that into zero-length requests.
        qsl = torch.full((self.max_batch + 1,), -1, dtype=torch.int32)
        acc = 0
        qsl[0] = 0
        for i, ql in enumerate(q_lens):
            acc += ql
            qsl[i + 1] = acc
        seq_lens = torch.zeros(self.max_batch, dtype=torch.int32)
        seq_lens[:b] = torch.tensor(kv_lens, dtype=torch.int32)
        self.qsl_buf.copy_(qsl.npu())
        self.seq_lens_buf.copy_(seq_lens.npu())

        # KV: quantize bf16 sources and scatter into the paged caches exactly
        # as forward() does -- mx quant for K, static per-channel for V.
        total_kv = sum(kv_lens)
        k_bf16 = (torch.randn(total_kv, self.nkv, self.d) * 0.5).bfloat16().npu()
        v_bf16 = (torch.randn(total_kv, self.nkv, self.d) * 0.5).bfloat16().npu()
        k_fp8, k_scale = torch_npu.npu_dynamic_mx_quant(k_bf16, dst_type=FP8_DTYPE)
        v_flat = v_bf16.view(total_kv, -1)
        v_fp8 = torch_npu.npu_quantize(v_flat, self.v_recip, None, FP8_DTYPE, -1, False)
        v_fp8 = v_fp8.view(total_kv, self.nkv, self.d)

        slots = []
        for i, kv in enumerate(kv_lens):
            base = i * self.max_blocks_per_req * self.block_size
            slots.append(torch.arange(base, base + kv, dtype=torch.int32))
        slot_mapping = torch.cat(slots).npu()

        self.mxfp.scatter_mxfp_pa_nz_kv_cache(
            k_fp8, v_fp8, self.k_cache, self.v_cache, slot_mapping, self.block_size)
        self.mxfp.scatter_mxfp_k_scale_cache(
            k_scale.view(torch.uint8) if k_scale.dtype != torch.uint8 else k_scale,
            self.k_scale_cache,
            self.mxfp.mxfp_k_scale_slot_index(slot_mapping, self.block_size),
        )

        q_bf16 = (torch.randn(num_tokens, self.nq, self.d) * 0.5).bfloat16().npu()
        q_fp8, q_scale = torch_npu.npu_dynamic_mx_quant(q_bf16, dst_type=FP8_DTYPE)
        return num_tokens, q_fp8, q_scale

    def step_lengths(self, num_tokens: int):
        """Derive (cu_seqlens_q, seqused_kv) on device, as _qfa_step_lengths does."""
        return (
            self.qsl_buf.clamp(min=0, max=num_tokens).cummax(dim=0).values,
            self.seq_lens_buf.clamp(min=1),
        )

    def v_descale_placeholder(self):
        """_qfa_v_descale_placeholder: the first two bytes of the V scale cache."""
        return as_e8m0(self.v_scale_cache.view(-1)[:2].view(1, 1, 1, 1, 1, 2))

    def caches_for_op(self):
        k = nz_5d_view(self.k_cache, self.nkv, self.d, self.block_size).view(FP8_DTYPE)
        v = nz_5d_view(self.v_cache, self.nkv, self.d, self.block_size).view(FP8_DTYPE)
        return k, v, as_e8m0(self.k_scale_cache), as_e8m0(self.v_scale_cache)


# --------------------------------------------------------------------------
# Call wrappers (argument sets copied from _get_qfa_metadata / _run_qfa)
# --------------------------------------------------------------------------
def build_metadata(metadata_op, batch: Batch, *, cu_seqlens_q, seqused_kv,
                   max_seqlen_q, mask_mode, layout_q_descale, v_descale=None):
    return metadata_op(
        batch.nq, batch.nkv, batch.d, QUANT_MODE_MXFP8,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=None,
        seqused_q=None,
        seqused_kv=seqused_kv,
        v_descale=batch.v_descale_placeholder() if v_descale is None else v_descale,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=-1,
        mask_mode=mask_mode,
        win_left=-1,
        win_right=-1,
        layout_q=LAYOUT_TND,
        layout_q_descale=layout_q_descale,
        layout_kv=LAYOUT_PA_NZ,
        layout_out=LAYOUT_TND,
    )


def run_main(main_op, batch: Batch, *, q_fp8, q_descale, metadata, cu_seqlens_q,
             seqused_kv, max_seqlen_q, mask_mode, layout_q_descale):
    k, v, k_scale, v_scale = batch.caches_for_op()
    result = main_op(
        q_fp8, k, v, as_e8m0(q_descale), k_scale, v_scale, QUANT_MODE_MXFP8,
        block_table=batch.block_table,
        p_scale=None,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=None,
        seqused_q=None,
        seqused_kv=seqused_kv,
        sinks=None,
        attn_mask=(None if mask_mode == MASK_MODE_NO_MASK else batch.mask),
        metadata=metadata,
        softmax_scale=batch.softmax_scale,
        mask_mode=mask_mode,
        win_left=-1,
        win_right=-1,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=-1,
        layout_q=LAYOUT_TND,
        layout_q_descale=layout_q_descale,
        layout_kv=LAYOUT_PA_NZ,
        layout_out=LAYOUT_TND,
        return_softmax_lse=False,
    )
    out = result[0] if isinstance(result, tuple) else result
    return out


def plan_for(batch: Batch, q_lens: list[int], kv_lens: list[int]):
    """Everything one step needs, derived the way forward() derives it."""
    num_tokens, q_fp8, q_scale_tnd = batch.load(q_lens, kv_lens)
    cu_q, seqused_kv = batch.step_lengths(num_tokens)
    max_seqlen_q = max(q_lens)
    mask_mode = MASK_MODE_NO_MASK if max_seqlen_q == 1 else MASK_MODE_CAUSAL
    # _qfa_query_scale_for_layout's boundary: G * Q_S vs 80.
    group_size = batch.nq // batch.nkv
    if group_size * max_seqlen_q > 80:
        q_descale, layout = q_scale_tnd, LAYOUT_TND
    else:
        q_descale, layout = q_scale_tnd_to_n2tgd(q_scale_tnd, batch.nkv), LAYOUT_N2TGD
    return dict(num_tokens=num_tokens, q_fp8=q_fp8, q_descale=q_descale,
                cu_seqlens_q=cu_q, seqused_kv=seqused_kv,
                max_seqlen_q=max_seqlen_q, mask_mode=mask_mode,
                layout_q_descale=layout)


def sane_output(name: str, out: torch.Tensor, num_tokens: int, nq: int, d: int) -> bool:
    """No golden here, so: right shape, finite, and not the all-zero output that
    a zeroed descale cache produces (the MTP-accept-rate collapse signature)."""
    host = out.float().cpu()
    shape_ok = tuple(out.shape) in {(num_tokens, nq, d), (num_tokens, nq * d)}
    finite = bool(torch.isfinite(host).all())
    nonzero = float(host.abs().max()) > 0.0
    print(f"  [{name}] shape={tuple(out.shape)} shape_ok={shape_ok} finite={finite} "
          f"max_abs={float(host.abs().max()):.6f} mean_abs={float(host.abs().mean()):.6f}")
    return shape_ok and finite and nonzero


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------
def case_sig(ctx) -> bool:
    """Hard gate: is every keyword attention_v1.py passes still there?"""
    main_op, metadata_op = ctx["ops"]
    ok = True
    for label, fn, expected in (("metadata", metadata_op, METADATA_KWARGS),
                                ("main", main_op, MAIN_KWARGS)):
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError) as exc:
            print(f"  [{label}] signature unavailable: {exc} -- cannot gate, "
                  f"treating as PASS and letting the calls speak")
            continue
        params = sig.parameters
        var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        missing = [k for k in expected if k not in params]
        print(f"  [{label}] {len(params)} params, **kwargs={var_kw}")
        print(f"  [{label}] signature: {sig}")
        if missing and not var_kw:
            print(f"  [{label}] MISSING keywords attention_v1.py passes: {missing}")
            ok = False
        elif missing:
            print(f"  [{label}] not named explicitly (absorbed by **kwargs): {missing}")
    return ok


def case_decode(ctx) -> bool:
    main_op, metadata_op = ctx["ops"]
    batch = ctx["batch"]
    step = plan_for(batch, [1] * 4, [300, 1025, 512, 4096])
    print(f"  Q_S=1 B=4 mask_mode={step['mask_mode']} "
          f"layout_q_descale={step['layout_q_descale']}")
    md = build_metadata(metadata_op, batch, **_md_args(step))
    out = run_main(main_op, batch, metadata=md, **_main_args(step))
    torch.npu.synchronize()
    return sane_output("DECODE", out, step["num_tokens"], batch.nq, batch.d)


def case_prefill(ctx) -> bool:
    main_op, metadata_op = ctx["ops"]
    batch = ctx["batch"]
    q_lens = [1024, 512, 37]
    step = plan_for(batch, q_lens, q_lens)
    print(f"  varlen q={q_lens} mask_mode={step['mask_mode']} "
          f"layout_q_descale={step['layout_q_descale']}")
    md = build_metadata(metadata_op, batch, **_md_args(step))
    out = run_main(main_op, batch, metadata=md, **_main_args(step))
    torch.npu.synchronize()
    return sane_output("PREFILL", out, step["num_tokens"], batch.nq, batch.d)


def case_plan_size(ctx) -> bool:
    """Does the plan's size depend on the tensor inputs, or only on the attrs?

    attention_v1.py keys its per-step plan cache on the non-tensor inputs only
    and, under capture, allocates the plan once. Both assume the size is a
    function of the attrs. Descriptive, not RED: dynamic is not itself a bug,
    it just invalidates two pieces of reasoning in the engine.
    """
    _, metadata_op = ctx["ops"]
    batch = ctx["batch"]
    probes = [
        ("decode B=1",       [1] * 1,  [512]),
        ("decode B=4",       [1] * 4,  [300, 1025, 512, 4096]),
        ("decode B=16",      [1] * 16, [512] * 16),
        ("decode B=16 long", [1] * 16, [8192] * 16),
        ("prefill 1x512",    [512],    [512]),
        # Same attrs as the row above, twice the batch: the prefill side's
        # own static-vs-dynamic pair, since that is the side PD-P runs.
        ("prefill 2x512",    [512, 512], [512, 512]),
        ("prefill 3xvarlen", [1024, 512, 37], [1024, 512, 37]),
        ("prefill 1x8192",   [8192],   [8192]),
    ]
    seen = {}
    rows = []
    for label, q_lens, kv_lens in probes:
        if sum(q_lens) > ctx["max_tokens"] or max(kv_lens) > ctx["max_kv"]:
            rows.append((label, "skipped (exceeds allocated capacity)", None))
            continue
        step = plan_for(batch, q_lens, kv_lens)
        md = build_metadata(metadata_op, batch, **_md_args(step))
        torch.npu.synchronize()
        desc = f"shape={tuple(md.shape)} dtype={md.dtype} numel={md.numel()}"
        rows.append((label, desc, md.numel()))
        key = (step["max_seqlen_q"], step["mask_mode"], step["layout_q_descale"])
        seen.setdefault(key, set()).add(md.numel())
    print("  plan size per probe:")
    for label, desc, _ in rows:
        print(f"    {label:22s} {desc}")
    # Same attrs, different batch/lengths -> did numel move?
    dynamic = [k for k, sizes in seen.items() if len(sizes) > 1]
    all_sizes = {n for _, _, n in rows if n is not None}
    if dynamic:
        print(f"  [PLAN-SIZE] DYNAMIC: plan numel varies with the tensor inputs "
              f"at fixed attrs {dynamic}")
        print("  [PLAN-SIZE] => attention_v1.py's per-step plan cache key "
              "(_get_qfa_metadata) is incomplete, and a captured graph would "
              "freeze the plan at its capture-time size. Both need revisiting.")
    elif len(all_sizes) > 1:
        print(f"  [PLAN-SIZE] attr-dependent only (sizes {sorted(all_sizes)}): "
              "the plan size tracks max_seqlen_q / mask_mode / layout, which the "
              "cache key already covers.")
    else:
        print(f"  [PLAN-SIZE] STATIC: one size {sorted(all_sizes)} everywhere.")
    return True


def case_plan_ro(ctx) -> bool:
    """One plan, three consecutive main-op calls: outputs and plan bytes stable?"""
    main_op, metadata_op = ctx["ops"]
    batch = ctx["batch"]
    step = plan_for(batch, [1] * 4, [300, 1025, 512, 4096])
    md = build_metadata(metadata_op, batch, **_md_args(step))
    torch.npu.synchronize()
    before = md.cpu().clone()
    outs = []
    for _ in range(3):
        outs.append(run_main(main_op, batch, metadata=md, **_main_args(step)).cpu().clone())
    torch.npu.synchronize()
    after = md.cpu()
    plan_stable = torch.equal(before, after)
    out_stable = all(torch.equal(outs[0], o) for o in outs[1:])
    print(f"  plan bytes unchanged after 3 calls={plan_stable}")
    print(f"  outputs bit-exact across 3 calls={out_stable}")
    if not plan_stable:
        print("  [PLAN-RO] the main op WRITES BACK into the plan -- sharing one "
              "plan across a step's 23 layers (_get_qfa_metadata) is unsafe: "
              "layer 1 right, layers 2..N quietly wrong.")
    return plan_stable and out_stable


def case_vdesc(ctx) -> bool:
    """Is the 6-D 1-element v_descale placeholder still accepted and inert?"""
    main_op, metadata_op = ctx["ops"]
    batch = ctx["batch"]
    step = plan_for(batch, [1] * 4, [300, 1025, 512, 4096])
    md_args = _md_args(step)
    try:
        md_ph = build_metadata(metadata_op, batch, **md_args)
        torch.npu.synchronize()
    except Exception as exc:
        print(f"  placeholder REJECTED by the metadata op: {type(exc).__name__}: {exc}")
        print("  [VDESC] _qfa_v_descale_placeholder no longer satisfies the "
              "checker -- attention_v1.py needs a real-shaped v_descale.")
        return False
    print(f"  placeholder accepted: shape={tuple(batch.v_descale_placeholder().shape)}")
    md_full = build_metadata(metadata_op, batch,
                             v_descale=as_e8m0(batch.v_scale_cache), **md_args)
    torch.npu.synchronize()
    out_ph = run_main(main_op, batch, metadata=md_ph, **_main_args(step)).cpu()
    out_full = run_main(main_op, batch, metadata=md_full, **_main_args(step)).cpu()
    torch.npu.synchronize()
    same = torch.equal(out_ph, out_full)
    print(f"  placeholder vs full-size v_descale bit-exact={same}")
    if not same:
        print("  [VDESC] the plan DOES depend on v_descale under PA_NZ -- the "
              "placeholder is not inert any more.")
    return same


def _graph_case(ctx, name, q_lens_a, kv_lens_a, q_lens_b, kv_lens_b) -> bool:
    """Capture with the metadata op INSIDE the compiled region (the engine's
    mode), swap the inputs in place, replay, compare against eager.

    The second batch deliberately keeps the same token total and batch size as
    the first: that is what a captured graph size means, and it is the only
    thing a replay is allowed to vary.
    """
    main_op, metadata_op = ctx["ops"]
    batch = ctx["batch"]
    try:
        import npugraph_ex  # noqa: F401
    except Exception as exc:
        print(f"  npugraph_ex unavailable ({type(exc).__name__}: {exc})")
        return None

    def step_fn():
        """One QFA step, metadata op included -- this whole thing gets captured."""
        cu_q, seqused_kv = batch.step_lengths(ctx["graph_tokens"])
        md = build_metadata(metadata_op, batch, cu_seqlens_q=cu_q,
                            seqused_kv=seqused_kv, max_seqlen_q=ctx["graph_max_q"],
                            mask_mode=ctx["graph_mask"],
                            layout_q_descale=ctx["graph_layout"])
        return run_main(main_op, batch, q_fp8=ctx["graph_q"], q_descale=ctx["graph_qd"],
                        metadata=md, cu_seqlens_q=cu_q, seqused_kv=seqused_kv,
                        max_seqlen_q=ctx["graph_max_q"], mask_mode=ctx["graph_mask"],
                        layout_q_descale=ctx["graph_layout"])

    # Batch A: eager reference, and the shapes the capture is built around.
    step_a = plan_for(batch, q_lens_a, kv_lens_a)
    ctx.update(graph_tokens=step_a["num_tokens"], graph_max_q=step_a["max_seqlen_q"],
               graph_mask=step_a["mask_mode"], graph_layout=step_a["layout_q_descale"])
    # Fixed-address q buffers: the graph must see one address across replays.
    q_buf = step_a["q_fp8"].view(torch.uint8).clone()
    qd_buf = (step_a["q_descale"].view(torch.uint8)
              if step_a["q_descale"].dtype != torch.uint8 else step_a["q_descale"]).clone()
    ctx["graph_q"] = q_buf.view(FP8_DTYPE)
    ctx["graph_qd"] = qd_buf
    print(f"  captured shape: T={step_a['num_tokens']} max_q={step_a['max_seqlen_q']} "
          f"mask_mode={step_a['mask_mode']} layout={step_a['layout_q_descale']}")

    ref_a = step_fn().cpu().clone()
    torch.npu.synchronize()

    compiled = torch.compile(step_fn, backend="npugraph_ex", dynamic=False)
    for _ in range(2):  # warmup + capture
        out = compiled()
    torch.npu.synchronize()
    replay_a = out.cpu().clone()
    exact_a = torch.equal(replay_a, ref_a)
    print(f"  [A] replay(batch A) vs eager bit-exact={exact_a}")

    # Batch B: refill every buffer in place, keeping T and B, then replay.
    step_b = plan_for(batch, q_lens_b, kv_lens_b)
    assert step_b["num_tokens"] == step_a["num_tokens"], "graph case needs equal T"
    assert step_b["layout_q_descale"] == step_a["layout_q_descale"]
    q_buf.copy_(step_b["q_fp8"].view(torch.uint8))
    qd_buf.copy_(step_b["q_descale"].view(torch.uint8)
                 if step_b["q_descale"].dtype != torch.uint8 else step_b["q_descale"])
    ref_b = step_fn().cpu().clone()
    torch.npu.synchronize()
    out_b = compiled()
    torch.npu.synchronize()
    exact_b = torch.equal(out_b.cpu(), ref_b)
    print(f"  [B] replay(batch B, buffers swapped in place) vs eager "
          f"bit-exact={exact_b}")
    if not exact_b:
        print(f"  [{name}] replay did not follow the refreshed lengths -- "
              "something in the step froze at capture time.")
    return exact_a and exact_b


def case_graph_decode(ctx):
    return _graph_case(ctx, "GRAPH-DECODE",
                       [1] * 4, [300, 1025, 512, 4096],
                       [1] * 4, [4096, 300, 1025, 512])


def case_graph_prefill(ctx):
    return _graph_case(ctx, "GRAPH-PREFILL",
                       [1024, 512, 37], [1024, 512, 37],
                       [512, 1024, 37], [512, 1024, 37])


def _md_args(step):
    return dict(cu_seqlens_q=step["cu_seqlens_q"], seqused_kv=step["seqused_kv"],
                max_seqlen_q=step["max_seqlen_q"], mask_mode=step["mask_mode"],
                layout_q_descale=step["layout_q_descale"])


def _main_args(step):
    return dict(q_fp8=step["q_fp8"], q_descale=step["q_descale"],
                cu_seqlens_q=step["cu_seqlens_q"], seqused_kv=step["seqused_kv"],
                max_seqlen_q=step["max_seqlen_q"], mask_mode=step["mask_mode"],
                layout_q_descale=step["layout_q_descale"])


CASES = {
    "SIG": case_sig,
    "DECODE": case_decode,
    "PREFILL": case_prefill,
    "PLAN-SIZE": case_plan_size,
    "PLAN-RO": case_plan_ro,
    "VDESC": case_vdesc,
    "GRAPH-DECODE": case_graph_decode,
    "GRAPH-PREFILL": case_graph_prefill,
}


# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="2.4t", choices=sorted(MODELS),
                        help="per-rank head shape (default 2.4t)")
    parser.add_argument("--cases", default="all",
                        help=f"comma-separated subset of {','.join(CASES)}")
    parser.add_argument("--worktree", default=os.environ.get("QFA_WORKTREE"),
                        help="vllm-ascend worktree to load mxfp_kv_cache.py from")
    parser.add_argument("--max-kv", type=int, default=8192,
                        help="KV capacity to allocate per request (default 8192)")
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="largest query token total any case may build")
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    names = list(CASES) if args.cases == "all" else [c.strip().upper()
                                                    for c in args.cases.split(",")]
    unknown = [n for n in names if n not in CASES]
    if unknown:
        parser.error(f"unknown case(s) {unknown}; known: {','.join(CASES)}")

    shape = MODELS[args.model]
    print("=== QFA delivered-operator test (cann_ops_transformer, PA_NZ) ===")
    print(f"model={args.model} {shape} group_size={shape['num_heads'] // shape['num_kv_heads']}")

    try:
        import torch_npu
        print(f"torch={torch.__version__} torch_npu={torch_npu.__version__}")
    except Exception as exc:
        print(f"[RED] torch_npu unavailable: {exc}")
        return 1
    try:
        ops = get_qfa_ops()
    except Exception as exc:
        print(f"[RED] cannot import cann_ops_transformer.ops."
              f"quant_flash_attn(_metadata): {type(exc).__name__}: {exc}")
        print("      This is the package the junlin-c8-mxfp* branches call. "
              "Without it every case below is meaningless.")
        return 1
    mxfp, source = load_mxfp_helpers(args.worktree)
    print(f"layout helpers from: {source}")

    max_blocks = -(-args.max_kv // shape["block_size"])
    batch = Batch(shape, max_blocks, args.max_batch, args.seed)
    batch.allocate(mxfp)
    print(f"allocated: num_blocks={batch.num_blocks} block_size={batch.block_size} "
          f"max_blocks_per_req={max_blocks} k_scale{tuple(batch.k_scale_cache.shape)} "
          f"v_scale{tuple(batch.v_scale_cache.shape)}")

    ctx = dict(ops=ops, batch=batch, mxfp=mxfp,
               max_kv=max_blocks * shape["block_size"], max_tokens=args.max_tokens)

    results = {}
    for name in names:
        print(f"\n== {name} ==")
        try:
            verdict = CASES[name](ctx)
        except Exception:
            traceback.print_exc()
            verdict = False
        if verdict is None:
            results[name] = "SKIP"
        else:
            results[name] = "GREEN" if verdict else "RED"
        print(f"  [{name}] {results[name]}")
        if name == "SIG" and results[name] == "RED":
            print("\n[RED] signature gate failed -- the delivered operator no "
                  "longer takes what attention_v1.py passes. Fix the call site "
                  "before reading anything else.")
            break

    print("\n=== summary ===")
    for name in names:
        print(f"  {name:14s} {results.get(name, 'not run')}")
    reds = [n for n, v in results.items() if v == "RED"]
    print(f"\n[{'RED' if reds else 'GREEN'}] "
          f"{len(reds)} RED, {sum(1 for v in results.values() if v == 'SKIP')} SKIP")
    return 1 if reds else 0


if __name__ == "__main__":
    sys.exit(main())
