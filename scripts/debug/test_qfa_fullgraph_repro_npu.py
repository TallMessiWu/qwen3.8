#!/usr/bin/env python3
"""Reproduce the FULL-graph QuantFlashAttn crash outside the engine.

Why another graph-capture script. test_qfa_graph_capture_npu.py's
ENGINE-INPOOL / ENGINE-OUTPOOL are cited as having cleared "the engine's
structure", but they cleared it for code that no longer exists: they lift
`_qfa_quant` / `_qfa_quant_v` out of attention_v1.py and quantize a bf16 cache
inside the capture. `c85c545ec` deleted both helpers along with the whole bf16
QFA path, so on junlin-qfa-graph that script cannot even bootstrap. What the
engine runs now is `_qfa_paged_call`: a four-plane MXFP8 cache read straight,
with only q quantized. Those green lights also ran at max_seqlen_kv=2048,
block_size=128, num_kv_heads=4 and a three-column block table -- none of which
is what the server runs.

So this rebuilds the same structure -- L layers sharing one capture-owned
buffer set per graph size, several sizes in one pool, every layer's event
released by a single refresh, the plan refreshed before replay -- against the
current op arrangement and the serving shapes, then sweeps one dimension at a
time. Mirrors, function for function:

  full_graph_qfa            -> capture_size()        buffers allocated inside
                               the capture and left empty, one event per layer
  _qfa_paged_call           -> qfa_call()            q quantized, K/V and both
                               scale planes read from the cache untouched
  _attach_qfa_inputs        -> DecodeStep.plan()     one AICPU plan per step,
                               placeholder v_descale, constant max_seqlen_kv
  _update_qfa_graph_buffers -> refresh_and_replay()  wait on the plan's event,
                               copy four buffers, record every layer's event

Cases -- each changes exactly one thing against REAL:

  EAGER-REF    no capture at all. The server serves fine eagerly at these
               shapes, so RED here means the harness is wrong, not the engine.
  REAL         serving shapes + engine structure. The main reproduction.
  MAXKV-2K     max_seqlen_kv 133120 -> 2048, the value ENGINE-* passed. RED on
               REAL and GREEN here puts it on the baked constant.
  BLOCK-128    block_size 512 -> 128, which also regroups the V scale plane.
  TABLE-TIGHT  block table narrowed from ceil(max_model_len/block_size) columns
               to just enough for kv_len. Asks whether the op takes a stride
               from max_seqlen_kv instead of from the tensor it is handed.
  KVLEN-SHORT  kv_len 1553 -> 300, ENGINE-*'s value.
  POOL-FAT     --pool-pad-mb of live padding allocated inside the capture,
               between the buffers and q/out. The plog has every core stopped
               at one PC with a fault address in the graph pool about 248MB
               past the furthest buffer; a small pool may not reach far enough
               to fault at all.
  SIZES-ALL    every capture size 27B.sh passes, sharing one pool.
  INGRAPH-CACHE the engine's real per-layer order: this step's K/V quantized,
               reshape_and_cache and scatter_mxfp_k_scale_cache writing the
               cache, then QFA reading that same cache -- all inside the one
               capture. Write-then-read of the same memory in one graph is
               the last structural difference the cases above leave out.
  COMPILED     the QFA call wrapped in torch.compile(backend="eager") before
               capture. vllm-ascend sets use_inductor=False, so eager-backend
               Dynamo is the honest approximation of the compiled region -- an
               approximation, not the engine's VllmBackend.

Every case prints the data_ptr of each buffer, q and the output, so the pool
layout here can be held against the plog's `[AIC_INFO] args after execute`
table and that 248MB gap. The layout matches it: the quantized q, its descale
and the output are allocated inside the capture and so land in the pool, the
four plane caches and the mask outside it.

Two deliberate simplifications, neither of which moves the pool: every layer
shares one KV cache (the server gives each of its ten full-attention layers its
own, ~870MiB each), and the bf16 query is allocated outside the capture rather
than produced by an in-graph QKV projection. What the op is handed either way
is the fp8 query quantized inside the capture.

Each case runs in its own subprocess: a device abort poisons everything after
it, and a mis-ordered event hangs rather than fails, so --case-timeout applies.

Verdict: GREEN when every layer's replayed output is bit-exact against the same
call run eagerly, RED when it differs, crashes or times out.

Usage (in the serving container, on the junlin-qfa-graph checkout):
  python scripts/debug/test_qfa_fullgraph_repro_npu.py \
      --model-config /mnt/share/weight/Qwen3.5-35B-A3B-mxfp4-c8/config.json
  python scripts/debug/test_qfa_fullgraph_repro_npu.py --cases REAL
  python scripts/debug/test_qfa_fullgraph_repro_npu.py --cases POOL-FAT --pool-pad-mb 512

--model-config is required: num_heads is a shape the op tiles on, and guessing
it tests a model nobody serves. Pass --num-heads / --num-kv-heads / --head-dim
instead only to sweep a shape deliberately.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import copy
import importlib.util
import json
import math
import os
import pathlib
import subprocess
import sys

import torch

CASES = (
    "EAGER-REF",
    "REAL",
    "MAXKV-2K",
    "BLOCK-128",
    "TABLE-TIGHT",
    "KVLEN-SHORT",
    "POOL-FAT",
    "SIZES-ALL",
    "COMPILED",
    "INGRAPH-CACHE",
)

# 27B.sh's default CAPTURE_SIZES.
SERVING_SIZES = (
    "1,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,"
    "96,100,104,108,112,116,120,124,128"
)

# mxfp_kv_cache.py's constants, restated so a mismatch surfaces as a shape
# error here instead of as silent garbage.
MXFP8_GROUP_SIZE = 32
SCALE_GROUP_SIZE = 64
SCALE_VALUES_PER_GROUP = 2


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------
def find_attention_v1(override: str | None) -> pathlib.Path:
    if override:
        return pathlib.Path(override)
    spec = importlib.util.find_spec("vllm_ascend")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("vllm_ascend is not importable; pass --attention-v1")
    root = pathlib.Path(list(spec.submodule_search_locations)[0])
    return root / "attention" / "attention_v1.py"


def lift_quant_q(path: pathlib.Path):
    """Lift _qfa_quant_q out of attention_v1.py's source.

    Importing the module pulls in the plugin's whole graph and trips a circular
    import outside the order the engine happens to import things in. The helper
    is plain torch/torch_npu, so its source is lifted straight out and cannot
    drift from what the engine runs. The guard below fails loudly if it ever
    grows a dependency on the rest of the file.
    """
    import torch_npu

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_qfa_quant_q"]
    if not nodes:
        raise RuntimeError(f"{path} no longer defines _qfa_quant_q at module level")

    allowed = {"torch", "torch_npu"} | set(dir(builtins))
    for node in nodes:
        bound = {arg.arg for arg in node.args.args}
        bound |= {i.id for i in ast.walk(node) if isinstance(i, ast.Name) and isinstance(i.ctx, ast.Store)}
        for inner in ast.walk(node):
            if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Load):
                if inner.id not in bound and inner.id not in allowed:
                    raise RuntimeError(
                        f"_qfa_quant_q now references {inner.id!r} from attention_v1; lifting it "
                        "out no longer works -- import the module instead and fix the cycle"
                    )

    namespace: dict = {"torch": torch, "torch_npu": torch_npu}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    print(f"  lifted _qfa_quant_q from {path}")
    return namespace["_qfa_quant_q"]


def bootstrap(args):
    """Register the vendored ops, then hand back the plugin's own q quantizer."""
    import torch_npu  # noqa: F401

    try:
        from vllm_ascend.utils import bootstrap_custom_op_env

        bootstrap_custom_op_env(include_vendor_lib=True)
    except Exception:
        import vllm_ascend

        vendor = os.path.join(
            os.path.dirname(vllm_ascend.__file__), "_cann_ops_custom", "vendors", "custom_transformer"
        )
        if os.path.isdir(vendor):
            prev = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = vendor + (":" + prev if prev else "")
        else:
            print(f"  [WARN] custom opp vendor dir missing: {vendor}")
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    for name in ("npu_quant_flash_attn", "npu_quant_flash_attn_metadata"):
        assert hasattr(torch.ops._C_ascend, name), f"{name} not registered"

    return lift_quant_q(find_attention_v1(args.attention_v1))


HEAD_SHAPES = ("num_heads", "num_kv_heads", "head_dim")


def load_cache_writers():
    """The two cache writers the engine runs inside the graph, ahead of QFA.

    mxfp_kv_cache is plain torch/torch_npu and imports cleanly. DeviceOperator
    does not always -- device_op -> ops/__init__ -> moe_mlp -> device_op is a
    cycle outside the engine's import order -- so its K/V writer falls back to
    the op BaseDeviceAdaptor calls, which is what an A5 resolves to anyway.
    """
    import torch_npu
    from vllm_ascend.device.mxfp_kv_cache import scatter_mxfp_k_scale_cache

    try:
        from vllm_ascend.device.device_op import DeviceOperator

        write_kv = DeviceOperator.reshape_and_cache
        print("  cache writer: DeviceOperator.reshape_and_cache")
    except Exception as exc:  # noqa: BLE001
        print(f"  cache writer: npu_scatter_pa_kv_cache (DeviceOperator unimportable: {type(exc).__name__})")

        def write_kv(key, value, key_cache, value_cache, slot_mapping):
            torch_npu.npu_scatter_pa_kv_cache(
                key=key.contiguous(),
                value=value.contiguous(),
                key_cache=key_cache,
                value_cache=value_cache,
                slot_mapping=slot_mapping.contiguous(),
                cache_mode="Norm",
            )

    return write_kv, scatter_mxfp_k_scale_cache


class InGraphCacheWrite:
    """What the engine does inside the graph before QFA reads the cache.

    forward() quantizes this step's K and V, calls reshape_and_cache and
    scatter_mxfp_k_scale_cache, and only then reaches full_graph_qfa -- all in
    one captured region, writing and reading the same cache memory. The V scale
    plane is not written: save_v_scale_flag makes that a one-off at load time.

    slot_mapping is a capture-owned buffer for the same reason the lengths are:
    the graph fixes its address, and replay only refreshes the contents.
    """

    def __init__(self, args, step, layers: int, writers):
        self.args = args
        self.write_kv, self.scatter_k_scale = writers
        n, d = args.num_kv_heads, args.head_dim
        # One K/V per layer, as the engine has: each layer projects its own.
        self.keys = [
            torch.randn(step.num_tokens, n, d, dtype=torch.bfloat16, device="npu") for _ in range(layers)
        ]
        self.values = [torch.randn_like(k) for k in self.keys]
        # v_cache_scale_float_reciprocal, flat because the cache's V scale is
        # the neutral 127 -- see Cache.
        self.v_recip = torch.ones(n * d, dtype=torch.float32, device="npu")
        self.slot_buf: torch.Tensor | None = None

    def allocate(self, step) -> None:
        """Inside the capture, next to the other buffers.

        Zeroed rather than empty like the others: capture records without
        executing, so the contents never matter there -- but these are indices,
        and slot 0 is at least in range if anything ever does read them.
        """
        self.slot_buf = torch.zeros_like(step.slot_mapping())

    def run(self, layer: int, cache: Cache) -> None:
        """Inside the capture, once per layer, before that layer's QFA call."""
        import torch_npu

        key = self.keys[layer]
        key_mxfp8, key_scale = torch_npu.npu_dynamic_mx_quant(key, dst_type=torch.float8_e4m3fn)
        value = self.values[layer]
        value_mxfp8 = torch_npu.npu_quantize(
            value.view(value.shape[0], -1), self.v_recip, None, torch.float8_e4m3fn, -1, False
        )
        self.write_kv(
            key=key_mxfp8,
            value=value_mxfp8.view(value.shape),
            key_cache=cache.k,
            value_cache=cache.v,
            slot_mapping=self.slot_buf,
        )
        # Byte view: index_put_ on float8 errors or falls back to AICPU.
        self.scatter_k_scale(key_scale.view(torch.uint8), cache.k_scale, self.slot_buf, self.args.block_size)

    def refresh(self, step) -> None:
        """Before replay: new K/V and the slots this step writes them to."""
        self.slot_buf.copy_(step.slot_mapping())
        for key, value in zip(self.keys, self.values):
            key.copy_(torch.randn_like(key))
            value.copy_(torch.randn_like(value))


def apply_model_config(args) -> None:
    """Fill the head shapes from the checkpoint. No defaults -- they are the model's.

    A guessed head count tests a model nobody serves, and the op tiles on it, so
    a missing --model-config is an error rather than a fallback. Explicit flags
    still win, for sweeping a shape the checkpoint does not have.
    """
    given = {name: getattr(args, name) for name in HEAD_SHAPES if getattr(args, name) is not None}
    if not args.model_config:
        missing = [name for name in HEAD_SHAPES if name not in given]
        if missing:
            raise SystemExit(
                f"pass --model-config <checkpoint>/config.json, or give {['--' + m.replace('_', '-') for m in missing]} "
                "explicitly; these are the served model's shapes and there is no sane default"
            )
        print(f"  head shapes from flags only: {given}")
        return

    cfg = json.loads(pathlib.Path(args.model_config).read_text(encoding="utf-8"))
    # Qwen3.5 keeps the attention shapes under text_config on multimodal checkpoints.
    for key in ("text_config", "language_config", "llm_config"):
        if isinstance(cfg.get(key), dict):
            cfg = {**cfg, **cfg[key]}
    if args.num_heads is None:
        args.num_heads = int(cfg["num_attention_heads"])
    if args.num_kv_heads is None:
        args.num_kv_heads = int(cfg["num_key_value_heads"])
    if args.head_dim is None:
        head_dim = cfg.get("head_dim") or cfg["hidden_size"] // int(cfg["num_attention_heads"])
        args.head_dim = int(head_dim)
    overridden = {name: value for name, value in given.items()}
    print(
        f"  from {args.model_config}: num_heads={args.num_heads} "
        f"num_kv_heads={args.num_kv_heads} head_dim={args.head_dim}"
        + (f" (overridden on the command line: {overridden})" if overridden else "")
    )


# --------------------------------------------------------------------------
# the four-plane MXFP8 cache, in the layout mxfp_kv_cache.py builds
# --------------------------------------------------------------------------
def v_descale_stub() -> torch.Tensor:
    """What _attach_qfa_inputs hands the metadata op: non-null, PA_BBND's rank,
    minimal extents. Built through a uint8 view -- torch cannot fill a float8
    tensor directly."""
    return torch.zeros(1, 1, 1, 1, 2, dtype=torch.uint8).npu().view(torch.float8_e8m0fnu)


def causal_mask() -> torch.Tensor:
    # triu(diagonal=1) int8, 1 = masked future: what attention_v1 hands the op
    # and what mask_mode=3 expects. The doc example's tril is wrong.
    return torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()


class Cache:
    """One layer's C8-MXFP KV cache, in the shapes mxfp_kv_cache.py splits out.

    k / v            (num_blocks, block_size, num_kv_heads, head_dim)  fp8_e4m3
    k_scale          (num_blocks, block_size, num_kv_heads, D//64, 2)  e8m0
    v_scale          (num_blocks, block_size//64, num_kv_heads, D, 2)  e8m0

    K's scale groups along D, V's along the sequence -- that asymmetry is the
    cache's, not this script's. Both scale planes are kept as uint8 the way the
    engine stores them, and only become e8m0 at the call.

    V is filled with a flat scale rather than a quantizer's output because the
    engine does not quantize V either: kv_cache_type=K_DYNAMIC_V_STATIC loads a
    per-channel scale from the checkpoint and broadcasts it in once. 127 is the
    neutral value the loader falls back to.
    """

    def __init__(self, args, quant_q, num_blocks: int, seed: int):
        torch.manual_seed(seed)
        d, n, bs = args.head_dim, args.num_kv_heads, args.block_size
        dev = "npu"
        # Quantizing a bf16 draft is the only way to get a legal fp8 bit
        # pattern here: .to(float8_e4m3fn) has no NPU kernel, and random bytes
        # would include NaN/Inf encodings the op never sees in a real cache.
        k_ref = torch.randn(num_blocks, bs, n, d, dtype=torch.bfloat16, device=dev)
        k_fp8, k_scale = quant_q(k_ref, d)
        self.k = k_fp8
        self.k_scale = k_scale.view(torch.uint8).contiguous()
        del k_ref

        v_ref = torch.randn(num_blocks, bs, n, d, dtype=torch.bfloat16, device=dev)
        v_fp8, _ = quant_q(v_ref, d)
        self.v = v_fp8
        del v_ref
        self.v_scale = torch.full(
            (num_blocks, bs // SCALE_GROUP_SIZE, n, d, SCALE_VALUES_PER_GROUP),
            args.v_scale_fill,
            dtype=torch.uint8,
            device=dev,
        )

        expected_k = (num_blocks, bs, n, d // SCALE_GROUP_SIZE, SCALE_VALUES_PER_GROUP)
        assert tuple(self.k_scale.shape) == expected_k, (
            f"k_scale {tuple(self.k_scale.shape)} != mxfp_k_scale_cache_shape {expected_k}"
        )

    def rewrite(self, quant_q, args, seed: int) -> None:
        """Replace the cache contents in place, the way the next step would."""
        torch.manual_seed(seed)
        d = args.head_dim
        k_ref = torch.randn(*self.k.shape, dtype=torch.bfloat16, device="npu")
        k_fp8, k_scale = quant_q(k_ref, d)
        # fp8 through a uint8 view: copy_ on float8 either errors or falls back
        # to AICPU, which stalls the device.
        self.k.view(torch.uint8).copy_(k_fp8.view(torch.uint8))
        self.k_scale.copy_(k_scale.view(torch.uint8))
        v_ref = torch.randn(*self.v.shape, dtype=torch.bfloat16, device="npu")
        v_fp8, _ = quant_q(v_ref, d)
        self.v.view(torch.uint8).copy_(v_fp8.view(torch.uint8))

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.k, self.v, self.k_scale, self.v_scale))


class DecodeStep:
    """One decode step's QFA arguments, shared by every layer of that step."""

    def __init__(self, args, num_tokens: int, table_cols: int, num_blocks: int, seed: int):
        torch.manual_seed(seed)
        self.args = args
        self.batch = max(1, num_tokens // args.q_len)
        self.num_tokens = self.batch * args.q_len
        self.kv_lens = [args.kv_len] * self.batch
        dev = "npu"

        # Only the first ceil(kv_len/block_size) columns are ever read -- the
        # engine's table is allocated at max_model_len/block_size and the tail
        # is never touched. Unique block ids only where they matter keeps the
        # cache at the server's size instead of batch * 260 blocks.
        used = math.ceil(args.kv_len / args.block_size)
        table = torch.zeros(self.batch, table_cols, dtype=torch.int32)
        for i in range(self.batch):
            start = (i * used) % max(1, num_blocks - used)
            table[i, :used] = torch.arange(start, start + used, dtype=torch.int32)
        self.block_table = table.to(dev)

        self.cu_seqlens_q = torch.tensor(
            [0] + [(i + 1) * args.q_len for i in range(self.batch)], dtype=torch.int32, device=dev
        )
        self.seqused_kv = torch.tensor(self.kv_lens, dtype=torch.int32, device=dev)
        # actual_seq_lengths_q is cumulative, so its last entry is the token
        # count -- which is the graph size on captured steps.
        self.max_seqlen_q = self.num_tokens
        self.plan_ready = None
        self.metadata = self.plan()

    def op_kwargs(self, cu=None, seq=None) -> dict:
        return {
            "cu_seqlens_q": self.cu_seqlens_q if cu is None else cu,
            "seqused_kv": self.seqused_kv if seq is None else seq,
            "mask_mode": 3,
            "max_seqlen_q": self.max_seqlen_q,
            "max_seqlen_kv": self.args.max_seqlen_kv,
            "layout_q": "TND",
            "layout_q_descale": "TND",
            "layout_kv": "PA_BBND",
            "layout_out": "TND",
        }

    def plan(self) -> torch.Tensor:
        """The AICPU split plan, built the way _attach_qfa_inputs builds it.

        Records an event once the op is enqueued so a consumer on another
        stream can wait for the plan to actually be written -- the race
        253dd1d42 fixed.
        """
        metadata = torch.ops._C_ascend.npu_quant_flash_attn_metadata(
            self.args.num_heads,
            self.args.num_kv_heads,
            self.args.head_dim,
            1,
            v_descale=v_descale_stub(),
            **self.op_kwargs(),
        )
        self.plan_ready = torch.npu.current_stream().record_event()
        return metadata

    def advance(self, seed: int) -> None:
        """Shorten the sequences and rebuild the plan, as the next step would.

        Lengths change so the refreshed plan differs too; otherwise a replay
        check would only ever exercise the cache read.
        """
        torch.manual_seed(seed)
        self.kv_lens = [max(1, length - 37) for length in self.kv_lens]
        self.seqused_kv.copy_(torch.tensor(self.kv_lens, dtype=torch.int32))
        self.metadata = self.plan()

    def header(self) -> tuple:
        """(sectionNum, isFd, mBaseSize, s2BaseSize) -- quant_flash_attn_metadata.h."""
        return tuple(self.metadata[:4].tolist())

    def slot_mapping(self) -> torch.Tensor:
        """Where this step's K/V land: the last q_len positions of each sequence."""
        table = self.block_table.cpu()
        block_size = self.args.block_size
        slots = []
        for i in range(self.batch):
            for j in range(self.args.q_len):
                pos = max(0, self.kv_lens[i] - self.args.q_len + j)
                slots.append(int(table[i, pos // block_size]) * block_size + pos % block_size)
        return torch.tensor(slots, dtype=torch.int32, device="npu")


def qfa_call(args, quant_q, q, cache: Cache, block_table, metadata, op_kwargs, mask) -> torch.Tensor:
    """Mirrors AscendC8MXFPAttentionBackendImpl._qfa_paged_call.

    K, V and both scale planes go in untouched -- no quantization, no
    transpose. Only q is quantized, because the graph produces it in bf16 from
    this step's QKV projection and it exists nowhere else.
    """
    q_fp8, q_descale = quant_q(q, args.head_dim)
    out, _ = torch.ops._C_ascend.npu_quant_flash_attn(
        q_fp8,
        cache.k,
        cache.v,
        q_descale,
        cache.k_scale.view(torch.float8_e8m0fnu),
        cache.v_scale.view(torch.float8_e8m0fnu),
        1,
        block_table=block_table,
        attn_mask=mask,
        metadata=metadata,
        softmax_scale=1.0 / math.sqrt(args.head_dim),
        **op_kwargs,
    )
    return out


# --------------------------------------------------------------------------
# the engine's capture / replay shape
# --------------------------------------------------------------------------
class Buffers:
    """What a captured QFA op reads: allocated by the capture, refreshed by it.

    QFA has no .out() variant and no task-group handle, so replay cannot rebind
    its parameters the way FIA does -- it reads these, and replay only refreshes
    their contents. One set per graph size serves every layer, because all of a
    step's full-attention layers share a single plan, block table and lengths.
    """

    def __init__(self, step: DecodeStep):
        self.cu_seqlens_q = torch.empty_like(step.cu_seqlens_q)
        self.seqused_kv = torch.empty_like(step.seqused_kv)
        self.metadata = torch.empty_like(step.metadata)
        self.block_table = torch.empty_like(step.block_table)

    def refresh(self, step: DecodeStep) -> list[tuple]:
        pairs = [
            (tuple(self.cu_seqlens_q.shape), tuple(step.cu_seqlens_q.shape)),
            (tuple(self.seqused_kv.shape), tuple(step.seqused_kv.shape)),
            (tuple(self.metadata.shape), tuple(step.metadata.shape)),
            (tuple(self.block_table.shape), tuple(step.block_table.shape)),
        ]
        self.cu_seqlens_q.copy_(step.cu_seqlens_q)
        self.seqused_kv.copy_(step.seqused_kv)
        self.metadata.copy_(step.metadata)
        self.block_table.copy_(step.block_table)
        return pairs

    def ptrs(self) -> str:
        return (
            f"cu=0x{self.cu_seqlens_q.data_ptr():x} seq=0x{self.seqused_kv.data_ptr():x} "
            f"meta=0x{self.metadata.data_ptr():x} bt=0x{self.block_table.data_ptr():x}"
        )


def make_layer_fn(args, quant_q, compiled: bool):
    """The per-layer call, optionally through Dynamo.

    vllm-ascend sets compilation_config.use_inductor = False, so the engine's
    compiled region is Dynamo + fx run eagerly, not inductor codegen. That is
    what backend="eager" reproduces. It is still an approximation: the engine
    runs vLLM's VllmBackend with its fusion passes over the whole model.
    """
    def layer(q, cache, block_table, metadata, op_kwargs, mask):
        return qfa_call(args, quant_q, q, cache, block_table, metadata, op_kwargs, mask)

    if not compiled:
        return layer
    return torch.compile(layer, backend="eager", dynamic=False)


def report_graph_breaks() -> None:
    """Say whether Dynamo actually traced anything.

    A COMPILED case that breaks on the custom op falls all the way back to
    eager, and would then report GREEN for having tested nothing.
    """
    try:
        from torch._dynamo.utils import counters
    except Exception:  # noqa: BLE001
        return
    breaks = sum(counters.get("graph_break", {}).values())
    print(
        f"    dynamo: {breaks} cumulative graph break(s)"
        + (" -- the layer traced into one graph" if not breaks else "")
    )


def capture_size(
    args, quant_q, cache: Cache, layer_fn, pool, size: int, table_cols: int, num_blocks: int, writers=None
):
    """Capture --layers QFA ops for one graph size, the way full_graph_qfa does."""
    step = DecodeStep(args, size, table_cols, num_blocks, args.seed + size)
    cache_write = InGraphCacheWrite(args, step, args.layers, writers) if writers else None
    qs = [
        torch.randn(step.num_tokens, args.num_heads, args.head_dim, dtype=torch.bfloat16, device="npu")
        for _ in range(args.layers)
    ]
    mask = causal_mask()

    if args.compiled:
        # Compile outside the capture: the engine warms up before capturing
        # too, and tracing inside a capture would record the guard evaluation.
        layer_fn(qs[0], cache, step.block_table, step.metadata, step.op_kwargs(), mask)
        torch.npu.synchronize()
        report_graph_breaks()

    graph = torch.npu.NPUGraph()
    events, outs = [], []
    padding = []
    context = torch.npu.graph(graph, pool=pool) if pool is not None else torch.npu.graph(graph)
    with context:
        # Taken inside the capture on purpose: torch.npu.graph runs the block on
        # a side stream, and only the capturing stream turns wait/reset into
        # graph nodes. Outside, the wait lands on the real default stream and
        # nothing releases it until replay -- the next capture's implicit
        # synchronize then hangs. The engine gets this right for free, because
        # full_graph_qfa is called from inside the captured forward.
        stream = torch.npu.current_stream()
        buffers = Buffers(step)
        if cache_write is not None:
            cache_write.allocate(step)
        if args.pool_pad_mb:
            # Held, not dropped: the point is to push q and the output far away
            # from the buffers inside the pool, the way a whole model's worth of
            # captured tensors does.
            padding.append(
                torch.empty(args.pool_pad_mb * 1024 * 1024, dtype=torch.uint8, device="npu")
            )
        for _ in range(args.layers):
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            events.append(event)
        # Left empty on purpose: filling them here would record the copies into
        # the graph, and at replay they would read capture-time sources that are
        # long gone.
        for index in range(args.layers):
            if cache_write is not None:
                # The engine's order: this step's K/V are quantized and written
                # into the cache, then QFA reads that same cache -- both inside
                # the graph.
                cache_write.run(index, cache)
            outs.append(
                layer_fn(
                    qs[index],
                    cache,
                    buffers.block_table,
                    buffers.metadata,
                    step.op_kwargs(cu=buffers.cu_seqlens_q, seq=buffers.seqused_kv),
                    mask,
                )
            )

    print(
        f"    size={size} batch={step.batch} header={step.header()} {buffers.ptrs()}\n"
        f"      q[0]=0x{qs[0].data_ptr():x} out[0]=0x{outs[0].data_ptr():x}"
        + (f" pad={args.pool_pad_mb}MB@0x{padding[0].data_ptr():x}" if padding else ""),
        flush=True,
    )
    return dict(step=step, qs=qs, buffers=buffers, graph=graph, events=events, outs=outs,
                mask=mask, padding=padding, cache_write=cache_write)


def refresh_and_replay(args, quant_q, cache: Cache, layer_fn, captured: dict, update_stream) -> bool:
    """Mirror _update_qfa_graph_buffers + replay, then check against eager."""
    step, qs, buffers = captured["step"], captured["qs"], captured["buffers"]
    graph, events, outs, mask = captured["graph"], captured["events"], captured["outs"], captured["mask"]
    cache_write = captured["cache_write"]

    # A different step, the way the next decode would be.
    cache.rewrite(quant_q, args, args.seed + step.num_tokens + 1000)
    for q in qs:
        q.copy_(torch.randn_like(q))
    step.advance(args.seed + step.num_tokens + 1000)
    if cache_write is not None:
        cache_write.refresh(step)

    # The eager reference runs the same in-graph cache writes, in the same
    # order, so both sides read a cache in the same state.
    refs = []
    for index, q in enumerate(qs):
        if cache_write is not None:
            cache_write.run(index, cache)
        refs.append(layer_fn(q, cache, step.block_table, step.metadata, step.op_kwargs(), mask).cpu())
    torch.npu.synchronize()
    # Proof the replay below actually re-runs the kernels rather than leaving
    # the capture-time outputs in place.
    before = outs[0].to(torch.float32).abs().sum().item()

    with torch.npu.stream(update_stream):
        # The AICPU plan op ran on the default stream; without this the copy can
        # outrun it and the captured op reads a half-written plan (253dd1d42).
        update_stream.wait_event(step.plan_ready)
        pairs = buffers.refresh(step)
        for event in events:
            event.record(update_stream)
    mismatched = [p for p in pairs if p[0] != p[1]]
    print(
        f"    replay size={step.num_tokens} header={step.header()} "
        f"copy shapes {'ok' if not mismatched else mismatched}",
        flush=True,
    )
    # What AclGraphWrapper does before replay in FULL mode.
    torch.npu.current_stream().synchronize()
    graph.replay()
    torch.npu.synchronize()

    after = outs[0].to(torch.float32).abs().sum().item()
    print(
        f"    replay rewrote out[0]: {before:.8g} -> {after:.8g}"
        + ("" if before != after else "  [WARN] unchanged -- did anything get captured?")
    )
    bad = [i for i in range(args.layers) if not torch.equal(outs[i].cpu(), refs[i])]
    print(f"    size={step.num_tokens}: {args.layers - len(bad)}/{args.layers} layers bit-exact"
          + (f", first bad={bad[0]}" if bad else ""))
    return not bad


def engine_repro(args, quant_q) -> bool:
    """Capture every --sizes graph into one pool, then replay each."""
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    table_cols = (
        math.ceil(args.kv_len / args.block_size)
        if args.tight_table
        else math.ceil(args.max_model_len / args.block_size)
    )
    max_batch = max(max(1, size // args.q_len) for size in sizes)
    num_blocks = max(args.blocks, max_batch * math.ceil(args.kv_len / args.block_size) + 1)

    cache = Cache(args, quant_q, num_blocks, args.seed)
    print(
        f"    cache {num_blocks} blocks x {args.block_size} x {args.num_kv_heads} x {args.head_dim} "
        f"= {cache.nbytes() / 2**20:.0f}MiB, table {table_cols} cols, max_seqlen_kv={args.max_seqlen_kv}\n"
        f"    k=0x{cache.k.data_ptr():x} v=0x{cache.v.data_ptr():x} "
        f"k_scale=0x{cache.k_scale.data_ptr():x} v_scale=0x{cache.v_scale.data_ptr():x}",
        flush=True,
    )

    layer_fn = make_layer_fn(args, quant_q, args.compiled)
    writers = load_cache_writers() if args.write_cache else None
    try:
        pool = torch.npu.graph_pool_handle()
        print(f"    shared graph pool: {pool}")
    except Exception as exc:  # noqa: BLE001
        pool = None
        print(f"    [WARN] no shared pool ({type(exc).__name__}); captures will not share memory")

    captured = {}
    for size in sizes:
        captured[size] = capture_size(
            args, quant_q, cache, layer_fn, pool, size, table_cols, num_blocks, writers
        )

    update_stream = torch.npu.Stream()
    ok = True
    for size in sizes:
        ok &= refresh_and_replay(args, quant_q, cache, layer_fn, captured[size], update_stream)
    return ok


def eager_ref(args, quant_q) -> bool:
    """No capture: the same shapes, run twice eagerly, checked against each other.

    The server serves at these shapes eagerly, so this is a harness check --
    if it is RED, nothing downstream of it means anything.
    """
    size = max(int(x) for x in args.sizes.split(",") if x.strip())
    table_cols = math.ceil(args.max_model_len / args.block_size)
    batch = max(1, size // args.q_len)
    num_blocks = max(args.blocks, batch * math.ceil(args.kv_len / args.block_size) + 1)

    cache = Cache(args, quant_q, num_blocks, args.seed)
    step = DecodeStep(args, size, table_cols, num_blocks, args.seed + size)
    mask = causal_mask()
    qs = [
        torch.randn(step.num_tokens, args.num_heads, args.head_dim, dtype=torch.bfloat16, device="npu")
        for _ in range(args.layers)
    ]
    print(f"    size={size} batch={batch} header={step.header()} table {table_cols} cols", flush=True)

    first = [qfa_call(args, quant_q, q, cache, step.block_table, step.metadata, step.op_kwargs(), mask).cpu()
             for q in qs]
    torch.npu.synchronize()
    second = [qfa_call(args, quant_q, q, cache, step.block_table, step.metadata, step.op_kwargs(), mask).cpu()
              for q in qs]
    torch.npu.synchronize()

    bad = [i for i in range(args.layers) if not torch.equal(first[i], second[i])]
    print(f"    {args.layers - len(bad)}/{args.layers} layers reproducible eagerly")
    return not bad


# --------------------------------------------------------------------------
# cases: each one changes exactly one thing against REAL
# --------------------------------------------------------------------------
def case_eager_ref(args, quant_q) -> bool:
    return eager_ref(args, quant_q)


def case_real(args, quant_q) -> bool:
    return engine_repro(args, quant_q)


def case_maxkv_2k(args, quant_q) -> bool:
    args = copy.copy(args)
    args.max_seqlen_kv = 2048
    return engine_repro(args, quant_q)


def case_block_128(args, quant_q) -> bool:
    args = copy.copy(args)
    args.block_size = 128
    return engine_repro(args, quant_q)


def case_table_tight(args, quant_q) -> bool:
    args = copy.copy(args)
    args.tight_table = True
    return engine_repro(args, quant_q)


def case_kvlen_short(args, quant_q) -> bool:
    args = copy.copy(args)
    args.kv_len = 300
    return engine_repro(args, quant_q)


def case_pool_fat(args, quant_q) -> bool:
    args = copy.copy(args)
    args.pool_pad_mb = args.pool_pad_mb or 256
    return engine_repro(args, quant_q)


def case_sizes_all(args, quant_q) -> bool:
    args = copy.copy(args)
    args.sizes = SERVING_SIZES
    return engine_repro(args, quant_q)


def case_ingraph_cache(args, quant_q) -> bool:
    args = copy.copy(args)
    args.write_cache = True
    return engine_repro(args, quant_q)


def case_compiled(args, quant_q) -> bool:
    args = copy.copy(args)
    args.compiled = True
    return engine_repro(args, quant_q)


RUNNERS = {
    "EAGER-REF": case_eager_ref,
    "REAL": case_real,
    "MAXKV-2K": case_maxkv_2k,
    "BLOCK-128": case_block_128,
    "TABLE-TIGHT": case_table_tight,
    "KVLEN-SHORT": case_kvlen_short,
    "POOL-FAT": case_pool_fat,
    "SIZES-ALL": case_sizes_all,
    "COMPILED": case_compiled,
    "INGRAPH-CACHE": case_ingraph_cache,
}


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default="EAGER-REF,REAL", help=f"comma-separated subset of {list(CASES)}")
    parser.add_argument("--case", help=argparse.SUPPRESS)  # child process entry
    parser.add_argument("--model-config", help="checkpoint config.json; required unless the shapes below are given")
    parser.add_argument("--num-heads", type=int, help="default: the checkpoint's")
    parser.add_argument("--num-kv-heads", type=int, help="default: the checkpoint's")
    parser.add_argument("--head-dim", type=int, help="default: the checkpoint's")
    parser.add_argument("--block-size", type=int, default=512, help="kernel block size (refresh_block_size)")
    parser.add_argument("--kv-len", type=int, default=1553, help="kv length per request")
    parser.add_argument("--q-len", type=int, default=1, help="tokens per request (1 + MTP drafts)")
    parser.add_argument("--layers", type=int, default=10, help="QFA ops per graph (full-attention layers)")
    parser.add_argument("--sizes", default="1,8,32,128", help="graph sizes sharing one pool")
    parser.add_argument("--blocks", type=int, default=1612, help="cache blocks, the server's count")
    parser.add_argument("--max-model-len", type=int, default=133120)
    parser.add_argument("--max-seqlen-kv", type=int, default=133120, help="the constant baked at capture")
    parser.add_argument("--v-scale-fill", type=int, default=127, help="static V scale byte (127 = neutral)")
    parser.add_argument("--pool-pad-mb", type=int, default=0, help="live padding held inside the capture")
    parser.add_argument("--tight-table", action="store_true", help="block table sized to kv_len only")
    parser.add_argument("--compiled", action="store_true", help="wrap the layer in torch.compile")
    parser.add_argument("--write-cache", action="store_true", help="write the cache inside the capture too")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention-v1", help="path to attention_v1.py (default: the installed one)")
    parser.add_argument("--case-timeout", type=float, default=900.0)
    return parser


def run_child(args) -> int:
    print(f"== {args.case} ==")
    try:
        apply_model_config(args)
        quant_q = bootstrap(args)
        ok = RUNNERS[args.case](args, quant_q)
    except Exception as exc:  # noqa: BLE001
        import traceback

        print(f"  [error] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        ok = False
    print(f"  [{args.case}] {'GREEN' if ok else 'RED'}")
    return 0 if ok else 1


def main() -> int:
    args = build_parser().parse_args()
    if args.case:
        return run_child(args)

    names = [n.strip() for n in args.cases.split(",") if n.strip()]
    unknown = [n for n in names if n not in RUNNERS]
    if unknown:
        print(f"[RED] unknown case(s) {unknown}; known: {list(RUNNERS)}")
        return 2
    # Resolve the head shapes once here so a missing --model-config is one
    # error, not the same error from every child.
    apply_model_config(copy.copy(args))

    passthrough: list[str] = []
    skip = False
    for arg in sys.argv[1:]:
        if skip:
            skip = False
        elif arg == "--cases":
            skip = True
        elif not arg.startswith("--cases="):
            passthrough.append(arg)

    results: dict[str, str] = {}
    for name in names:
        command = [sys.executable, os.path.abspath(__file__), "--case", name, *passthrough]
        try:
            completed = subprocess.run(command, timeout=args.case_timeout)
            results[name] = "GREEN" if completed.returncode == 0 else "RED"
        except subprocess.TimeoutExpired:
            print(f"  [{name}] TIMEOUT after {args.case_timeout}s -- likely a stalled event handshake")
            results[name] = "TIMEOUT"

    print("\n=== summary ===")
    for name in names:
        print(f"  {name:<14} {results[name]}")
    return 0 if all(v == "GREEN" for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
