#!/usr/bin/env python3
"""Does the FIA-call-site swap actually run on device?

attention_v1 now calls QuantFlashAttn where it used to call FIA. Two shapes to
check before starting a server, mirroring what the call site builds (the helpers
are copied rather than imported: importing attention_v1 on its own trips the
device_op <-> fused_moe circular import, which the engine avoids by registering
the platform first -- keep the two in step).

  dense  PrefillNoCache: this batch's own K/V, no block table, layout TND.
         The op rejects TND without cu_seqlens_kv ("cuSeqlensKvOptional should
         be provided"), and actual_seq_lengths_kv is cumulative in this state.
         The engine does not route this state to QFA -- attn_metadata.qfa is
         None under PrefillNoCache, so _qfa_serves says no -- which makes this
         case a test of the operator's TND layout rather than of a served
         path. --bench therefore has no dense row.
  paged  everything else, and everything the server actually runs: the cache
         arrives as (num_blocks, block_size, N*D), so the heads get split back
         out for PA_BBND, and seqused_kv is per-sequence here.

V is the odd one out in both: QFA groups its scales down the sequence
((T/64, N, D, 2) / (Bn, Bs/64, N, D, 2)), not along D like q and k.

Each case then runs FIA on the same bf16 inputs and compares. FIA is the
reference the swap has to reproduce; QFA reads an MXFP8 copy of the same K/V, so
the criterion is distributional (pass_rate + cosine, mirroring
result_compare_method) rather than bit-exact -- and the first token of a sequence
attends a single KV with no softmax averaging, so its pointwise error is large by
construction and says nothing about correctness.

Shapes come from --model (default 27b: Nq=24 Nkv=4 D=256 block 128; 35b:
Nq=16 Nkv=2 D=256 block 512), with per-field flags to override. Both models are
D=256, but the decode bandwidth ratio this measures depends on NKV*D bytes per
token and BLOCK per page -- measuring one model's numbers against the other's
server answers nothing. Each case runs in its own subprocess: an AICPU abort
poisons the device.

--bench times deployments instead of comparing them: C8 and QFA ship together,
so the number that decides anything is C16+FIA against C8+QFA with each side
carrying its own per-step costs. It also prints the two operators alone, on
inputs of different width -- an unfair comparison on purpose, because it is
what the attention layer's time actually becomes.

The two configurations 27B.sh can actually serve:

  C16+FIA  C8=0 (VLLM_ASCEND_DISABLE_C8_MXFP): bf16 cache, plain FIA. The
           baseline, and the only one an end-to-end comparison can use.
  C8+QFA   QFA=1: the MXFP8 cache read in place by QuantFlashAttn.

There is deliberately no third leg. "MXFP8 cache + FIA" would split the gap
into a bandwidth half and a kernel half, but it is not a configuration that
exists -- FIA dies with EZ0010 at head_dim 256, which is why QFA was vendored
in the first place.

Per-step per-layer costs only the C8 side pays:

  q-quant   the query is projected fresh in bf16 every step and QFA takes
            MXFP8. _qfa_paged_call quantizes q and nothing else.
  kv-quant  this step's new K/V, quantized before reshape_and_cache writes
            them. K is dynamic MXFP8; V rides the checkpoint's static
            per-channel scale. The read side touches neither: they are
            already FP8 in the cache, in the order PA_BBND wants.
  meta      the AICPU plan, paid once per step for every layer, so a
            per-layer number gets meta / attn_layers -- the full-attention
            count from the --model preset, since the stack is hybrid 3:1 and
            the linear layers never call QFA.

Scatter is not timed: both configurations scatter, and C8 scatters fewer
bytes, so charging it to C8 alone would flatter C16.

Every bench shape is paged. QFA only serves states that read the cache
(attn_metadata.qfa is None under PrefillNoCache), and with chunked prefill on
the FIA baseline reads it too -- _get_fia_params hands FIA the same
block_table. A dense TND prefill is a shape the engine does not run.

Attention decode is bound by KV bandwidth and the MXFP8 cache is roughly half
the bytes, so the interesting output is how the ratios move with context
length, not any single number.

Usage (inside the serving container, no server running):
  python scripts/bench/test_qfa_vs_fia.py
  python scripts/bench/test_qfa_vs_fia.py --case dense
  python scripts/bench/test_qfa_vs_fia.py --bench
  python scripts/bench/test_qfa_vs_fia.py --bench --attn-layers 15  # 397B stack
  python scripts/bench/test_qfa_vs_fia.py --model 35b
  python scripts/bench/test_qfa_vs_fia.py --model 35b --bench
  python scripts/bench/test_qfa_vs_fia.py --model 35b --all   # both halves
  python scripts/bench/test_qfa_vs_fia.py --bench --shape decode-b32-16k
"""

import argparse
import os
import subprocess
import sys

# Shape presets. Both served models are D=256, but the head counts and the
# kernel block size differ -- and the decode bandwidth ratio this script exists
# to measure depends on exactly those (NKV * D bytes per token, BLOCK per page),
# so measuring 27B's numbers against a 35B server would answer nothing.
# attn_layers is the full-attention count, which the AICPU plan is spread over:
# these are hybrid stacks, and only the full-attention layers run QFA at all.
# Both are 3:1 with full_attention_interval 4, so it is num_hidden_layers // 4 --
# Qwen/Qwen3.8-27B is 64 layers, 16 of them full attention. 35b is not in the
# Qwen3.8 line (which is 27B, Flash-Next and 2.4T-A95B); the served shape
# 16/2/256 is Qwen3.5-35B-A3B's, 40 layers and 10 full. Other stacks in this
# repo, if this script ever points at one: 397B-A17B 60/15, 2.4T-A95B 92/23.
MODELS = {
    "27b": {"num_heads": 24, "num_kv_heads": 4, "head_dim": 256,
            "block_size": 128, "prefill_len": 1594, "attn_layers": 16,
            "max_model_len": 133120},
    "35b": {"num_heads": 16, "num_kv_heads": 2, "head_dim": 256,
            "block_size": 512, "prefill_len": 1552, "attn_layers": 10,
            "max_model_len": 133120},
}

# Defaults are 27B's, as they always were. apply_shape() overwrites them from
# --model / the per-field flags before anything reads them; the module-level
# names stay so the call sites below need no threading.
NQ, NKV, D, BLOCK, WINDOW = 24, 4, 256, 128, 64
PREFILL_LEN = 1594
ATTN_LAYERS = 16
MAX_MODEL_LEN = 133120
DECODE_REQS, DECODE_KV = 4, 300

# WINDOW is MXFP8's scale grouping, not a model shape -- it stays 64.
# attn_layers and max_model_len ride along here because they come from --model
# and override the same way, though neither is a tensor shape: one is a stack
# depth, the other the constant QFA tiles on.
SHAPE_FLAGS = ("num_heads", "num_kv_heads", "head_dim", "block_size", "prefill_len",
               "attn_layers", "max_model_len")

# attention_v1's MXFP8_{QUERY,KEY,VALUE}_QUANT_MODE, for the C8+FIA leg. 6 is
# per-32-along-D, 8 is grouped down the sequence -- the same asymmetry QFA has.
MXFP8_QUERY_QUANT_MODE = 6
MXFP8_KEY_QUANT_MODE = 6
MXFP8_VALUE_QUANT_MODE = 8


def apply_shape(args) -> None:
    """Resolve the preset plus any per-field override into the globals."""
    global NQ, NKV, D, BLOCK, PREFILL_LEN, ATTN_LAYERS, MAX_MODEL_LEN
    preset = MODELS[args.model]
    resolved = {name: getattr(args, name) or preset[name] for name in SHAPE_FLAGS}
    NQ = resolved["num_heads"]
    NKV = resolved["num_kv_heads"]
    D = resolved["head_dim"]
    BLOCK = resolved["block_size"]
    PREFILL_LEN = resolved["prefill_len"]
    ATTN_LAYERS = resolved["attn_layers"]
    MAX_MODEL_LEN = resolved["max_model_len"]
    print(
        f"shapes: model={args.model} num_heads={NQ} num_kv_heads={NKV} "
        f"head_dim={D} block_size={BLOCK} prefill_len={PREFILL_LEN} "
        f"attn_layers={ATTN_LAYERS} max_model_len={MAX_MODEL_LEN}",
        flush=True,
    )


def shape_argv(args) -> list:
    """The shape flags, for passing down to a per-case subprocess."""
    argv = ["--model", args.model]
    for name in SHAPE_FLAGS:
        value = getattr(args, name)
        if value is not None:
            argv += ["--" + name.replace("_", "-"), str(value)]
    return argv
CASES = ["dense", "paged"]

# The shapes the live 27B config produces, all paged -- see the module
# docstring on why there is no dense row. Prefill is one request whose q is the
# chunk and whose kv is everything cached so far; 16384 is
# max_num_batched_tokens, the largest single chunk, and prefill_len is the
# multimodal prompt that first exercised this. Decode carries 1 + 3 MTP tokens
# per request, up to max_num_seqs=32, with kv running out to max_model_len.
#
# A function, not a constant: the row that uses prefill_len has to be built
# after apply_shape, or --model 35b would silently bench 27B's 1594.
def bench_shapes() -> list:
    return [
        ("prefill-512", 1, 512, 512),
        ("prefill-prompt", 1, PREFILL_LEN, PREFILL_LEN),
        ("prefill-4k", 1, 4096, 4096),
        ("prefill-16k", 1, 16384, 16384),
        ("decode-b32-1k", 32, 4, 1024),
        ("decode-b32-4k", 32, 4, 4096),
        ("decode-b32-16k", 32, 4, 16384),
        ("decode-b8-32k", 8, 4, 32768),
        ("decode-b1-128k", 1, 4, 131072),
    ]


BENCH_NAMES = [row[0] for row in bench_shapes()]


def dequant_along_d(fp8, scale, d):
    """Undo _qfa_quant: one e8m0 exponent per 32 elements along D."""
    import torch

    exp = scale.view(torch.uint8).reshape(*fp8.shape[:-1], d // 32).float() - 127.0
    return (fp8.float() * torch.pow(2.0, exp).repeat_interleave(32, dim=-1)).to(torch.bfloat16)


def dequant_along_seq(fp8, scale, seq_lens=None):
    """Undo quant_v_by_sequence: one exponent per 32 positions down the sequence.

    The trailing pair in the scale is the second half of each 64-wide window, so
    folding it into the sequence axis recovers one exponent per 32 rows. TND is
    quantized per sequence with the tail padded up to 64 and the fp8 trimmed back
    afterwards, so walk the sequences rather than expanding the whole thing.
    """
    import torch

    if seq_lens is None:  # PA_BBND (Bn, Bs, N, D), scale (Bn, Bs//64, N, D, 2)
        nb, w, n, d, _ = scale.shape
        exp = scale.view(torch.uint8).permute(0, 1, 4, 2, 3).reshape(nb, w * 2, n, d).float() - 127.0
        return (fp8.float() * torch.pow(2.0, exp).repeat_interleave(32, dim=1)).to(torch.bfloat16)

    n, d = fp8.shape[1], fp8.shape[2]  # TND (T, N, D), scale (sum ceil64(s), N, D, 2)
    parts, f_at, s_at = [], 0, 0
    for s_len in seq_lens:
        w = (s_len + 63) // 64
        sc = scale[s_at : s_at + w]
        s_at += w
        exp = sc.view(torch.uint8).permute(0, 3, 1, 2).reshape(w * 2, n, d).float() - 127.0
        full = torch.pow(2.0, exp).repeat_interleave(32, dim=0)[:s_len]
        parts.append(fp8[f_at : f_at + s_len].float() * full)
        f_at += s_len
    return torch.cat(parts).to(torch.bfloat16)


def compare(name, got, ref):
    """QFA (MXFP8 K/V) against FIA (bf16 K/V) on the same inputs.

    Accumulate in float64: cosine over ~10M float32 elements drifts enough to
    come back above 1.0. The two-per-mille criterion the single-op script uses
    does not apply here -- that one compares against a golden computed from the
    same quantized inputs, whereas this comparison carries the quantization loss
    itself, so judge by relative L2 and cosine and print the error spread.
    """
    import torch

    a = got.float().cpu().reshape(-1).double()
    b = ref.float().cpu().reshape(-1).double()
    diff = a - b
    rel_l2 = (diff.norm() / b.norm()).item()
    cos = (a @ b / (a.norm() * b.norm())).item()
    scale = b.abs().mean().item()
    q = torch.quantile(diff.abs(), torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64))
    print(
        f"  {name}: rel_l2={rel_l2:.5f} cos={cos:.6f} "
        f"|err| p50={q[0]:.5f} p90={q[1]:.5f} p99={q[2]:.5f} max={diff.abs().max():.4f} "
        f"(ref mean|x|={scale:.5f})",
        flush=True,
    )
    return rel_l2, cos


def quant_v_by_sequence(value, seq_lens=None):
    """Mirror of attention_v1._qfa_quant_v."""
    import torch
    import torch_npu

    fp8_dtype, e8m0 = torch.float8_e4m3fn, torch.float8_e8m0fnu
    if seq_lens is None:  # paged: (Bn, Bs, N, D) -> (Bn, Bs//64, N, D, 2)
        nb, bs, n, d = value.shape
        cols = value.permute(0, 2, 3, 1).contiguous().reshape(nb * n * d, bs)
        fp8, scale = torch_npu.npu_dynamic_mx_quant(cols, dst_type=fp8_dtype, scale_alg=0)
        return (
            fp8.view(torch.uint8).reshape(nb, n, d, bs).permute(0, 3, 1, 2).contiguous().view(fp8_dtype),
            scale.view(torch.uint8).reshape(nb, n, d, bs // WINDOW, 2)
            .permute(0, 3, 1, 2, 4).contiguous().view(e8m0),
        )
    n, d = value.shape[1], value.shape[2]     # TND: (T, N, D) -> (sum ceil64(s), N, D, 2)
    fp8_parts, scale_parts, start = [], [], 0
    for s in seq_lens:
        chunk = value[start: start + s]
        start += s
        s_pad = (s + WINDOW - 1) // WINDOW * WINDOW
        if s_pad != s:
            chunk = torch.nn.functional.pad(chunk, (0, 0, 0, 0, 0, s_pad - s))
        cols = chunk.permute(1, 2, 0).contiguous().reshape(n * d, s_pad)
        fp8, scale = torch_npu.npu_dynamic_mx_quant(cols, dst_type=fp8_dtype, scale_alg=0)
        fp8_parts.append(fp8.view(torch.uint8).reshape(n, d, s_pad).permute(2, 0, 1).contiguous()[:s])
        scale_parts.append(
            scale.view(torch.uint8).reshape(n, d, s_pad // WINDOW, 2).permute(2, 0, 1, 3).contiguous()
        )
    return torch.cat(fp8_parts).view(fp8_dtype), torch.cat(scale_parts).view(e8m0)


def run_case(name: str) -> int:
    import torch
    import torch_npu  # noqa: F401

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_qfa_op import bootstrap_ops

    torch.npu.set_device(int(os.environ.get("QFA_DEVICE", "0")))
    bootstrap_ops()

    def _qfa_quant(x, d):
        """Mirror of attention_v1._qfa_quant."""
        fp8, scale = torch_npu.npu_dynamic_mx_quant(
            x.reshape(-1, d), dst_type=torch.float8_e4m3fn, scale_alg=0)
        return (
            fp8.reshape(x.shape),
            scale.view(torch.uint8).reshape(*x.shape[:-1], d // 64, 2).view(torch.float8_e8m0fnu),
        )

    paged = name == "paged"
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()

    if paged:
        nb = 64
        q = torch.randn(DECODE_REQS, NQ, D, dtype=torch.bfloat16).npu()
        cache_k = torch.randn(nb, BLOCK, NKV * D, dtype=torch.bfloat16).npu()
        cache_v = torch.randn(nb, BLOCK, NKV * D, dtype=torch.bfloat16).npu()
        blocks_per_req = (DECODE_KV + BLOCK - 1) // BLOCK
        table = torch.arange(1, DECODE_REQS * blocks_per_req + 1, dtype=torch.int32)
        table = table.reshape(DECODE_REQS, blocks_per_req).npu()
        q_lens, kv_lens = [1] * DECODE_REQS, [DECODE_KV] * DECODE_REQS
        fia_key, fia_value = cache_k, cache_v  # FIA takes the cache view as-is
        k_fp8, k_descale = _qfa_quant(cache_k.reshape(nb, BLOCK, NKV, D), D)
        v_fp8, v_descale = quant_v_by_sequence(cache_v.reshape(nb, BLOCK, NKV, D))
        kv_args = {"seqused_kv": torch.tensor(kv_lens, dtype=torch.int32).npu()}
        layout_kv = "PA_BBND"
    else:
        q = torch.randn(PREFILL_LEN, NQ, D, dtype=torch.bfloat16).npu()
        key = torch.randn(PREFILL_LEN, NKV, D, dtype=torch.bfloat16).npu()
        value = torch.randn(PREFILL_LEN, NKV, D, dtype=torch.bfloat16).npu()
        table = None
        q_lens, kv_lens = [PREFILL_LEN], [PREFILL_LEN]
        fia_key, fia_value = key, value
        k_fp8, k_descale = _qfa_quant(key, D)
        v_fp8, v_descale = quant_v_by_sequence(value, kv_lens)
        cum_kv = []
        acc = 0
        for s_len in kv_lens:
            acc += s_len
            cum_kv.append(acc)
        kv_args = {"cu_seqlens_kv": torch.tensor([0] + cum_kv, dtype=torch.int32).npu()}
        layout_kv = "TND"

    q_fp8, q_descale = _qfa_quant(q, D)
    print(f"  q {tuple(q_fp8.shape)} qs {tuple(q_descale.shape)}", flush=True)
    print(f"  k {tuple(k_fp8.shape)} ks {tuple(k_descale.shape)}", flush=True)
    print(f"  v {tuple(v_fp8.shape)} vs {tuple(v_descale.shape)}", flush=True)

    cum = []
    acc = 0
    for s in q_lens:
        acc += s
        cum.append(acc)
    args = {
        "cu_seqlens_q": torch.tensor([0] + cum, dtype=torch.int32).npu(),
        "mask_mode": 3,
        "max_seqlen_q": max(q_lens),
        "max_seqlen_kv": max(kv_lens),
        "layout_q": "TND",
        "layout_q_descale": "TND",
        "layout_kv": layout_kv,
        "layout_out": "TND",
        **kv_args,
    }
    metadata = torch.ops._C_ascend.npu_quant_flash_attn_metadata(
        NQ, NKV, D, 1, v_descale=v_descale, **args)
    torch.npu.synchronize()
    print("  metadata ok", flush=True)
    out, _ = torch.ops._C_ascend.npu_quant_flash_attn(
        q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale, 1,
        block_table=table, attn_mask=mask, metadata=metadata,
        softmax_scale=D ** -0.5, **args)
    torch.npu.synchronize()
    print(f"  main op ok, out={tuple(out.shape)}", flush=True)

    # Same call with no plan. The doc calls metadata an optional scheduling
    # optimization; if that holds, the graph path can drop it and stop having to
    # keep two calls' arguments in step across capture and replay.
    try:
        out_nometa, _ = torch.ops._C_ascend.npu_quant_flash_attn(
            q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale, 1,
            block_table=table, attn_mask=mask, metadata=None,
            softmax_scale=D ** -0.5, **args)
        torch.npu.synchronize()
        nometa_ok = True
    except Exception as exc:  # noqa: BLE001 -- the answer is "it is not optional"
        print(f"  no-metadata call FAILED: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
        out_nometa, nometa_ok = None, False

    # Reference: the operator this call site used to use, same bf16 inputs.
    # attention_v1 passes get_splitfuse_attn_mask(), which is this same
    # triu(2048, diagonal=1) int8 -- so both operators see one mask.
    def run_fia(qq, kk, vv):
        out, _ = torch_npu.npu_fused_infer_attention_score(
            query=qq,
            key=kk,
            value=vv,
            atten_mask=mask,
            block_table=table,
            input_layout="TND",
            block_size=BLOCK,
            actual_seq_lengths=cum,
            actual_seq_lengths_kv=kv_lens if paged else cum,
            num_key_value_heads=NKV,
            num_heads=NQ,
            scale=D ** -0.5,
            sparse_mode=3,
        )
        torch.npu.synchronize()
        return out

    fia_out = run_fia(q, fia_key, fia_value)
    print(f"  fia out={tuple(fia_out.shape)}", flush=True)

    # Same operator, fed the dequantized tensors: isolates what QFA computes
    # from the loss of quantizing at all. QFA should land almost on top of this.
    q_deq = dequant_along_d(q_fp8, q_descale, D)
    k_deq = dequant_along_d(k_fp8, k_descale, D)
    if paged:
        v_deq = dequant_along_seq(v_fp8, v_descale).reshape(fia_key.shape)
        k_deq = k_deq.reshape(fia_key.shape)
    else:
        v_deq = dequant_along_seq(v_fp8, v_descale, kv_lens)
    fia_deq_out = run_fia(q_deq, k_deq, v_deq)

    l2_raw, cos_raw = compare("vs FIA(bf16)  ", out, fia_out)
    l2_deq, cos_deq = compare("vs FIA(deq K/V)", out, fia_deq_out)
    # The first comparison carries the quantization loss and is informational;
    # the second one is the verdict -- same inputs, same maths, so anything
    # beyond rounding means the two operators disagree about the layout.
    good = cos_deq >= 0.9995
    if nometa_ok:
        _, cos_nometa = compare("no-metadata   ", out_nometa, out)
        exact = torch.equal(out_nometa, out)
        print(f"  metadata dropped: bit-exact={exact} cos={cos_nometa:.6f}", flush=True)
        good = good and cos_nometa >= 0.9995
    print(f"  [{name}] quantization loss {l2_raw * 100:.1f}%, "
          f"operator agreement cos={cos_deq:.6f}, "
          f"no-metadata {'ok' if nometa_ok else 'REJECTED'} -> {'GREEN' if good else 'RED'}", flush=True)
    return 0 if good else 1


def _time(fn, iters: int, warmup: int) -> float:
    """Mean seconds per call, timed as a block so launch overhead amortizes."""
    import time

    import torch

    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) / iters


def run_bench(label: str, batch: int, q_len: int, kv_len: int,
              iters: int, warmup: int) -> int:
    """Time one attention layer, one step, in each of the two configurations."""
    import json

    import torch
    import torch_npu  # noqa: F401

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_qfa_op import bootstrap_ops

    torch.npu.set_device(int(os.environ.get("QFA_DEVICE", "0")))
    bootstrap_ops()

    def _qfa_quant(x, d):
        """Mirror of attention_v1._qfa_quant_q (see run_case)."""
        fp8, scale = torch_npu.npu_dynamic_mx_quant(
            x.reshape(-1, d), dst_type=torch.float8_e4m3fn, scale_alg=0)
        return (
            fp8.reshape(x.shape),
            scale.view(torch.uint8).reshape(*x.shape[:-1], d // 64, 2).view(torch.float8_e8m0fnu),
        )

    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()
    total_q = batch * q_len
    q = torch.randn(total_q, NQ, D, dtype=torch.bfloat16).npu()
    cum_q = [(i + 1) * q_len for i in range(batch)]
    kv_lens = [kv_len] * batch

    blocks_per_req = (kv_len + BLOCK - 1) // BLOCK
    nb = batch * blocks_per_req
    cache_k = torch.randn(nb, BLOCK, NKV * D, dtype=torch.bfloat16).npu()
    cache_v = torch.randn(nb, BLOCK, NKV * D, dtype=torch.bfloat16).npu()
    table = torch.arange(nb, dtype=torch.int32).reshape(batch, blocks_per_req).npu()

    # The C8 cache holds these already quantized; quantizing here just fills
    # the buffers QFA reads. Not timed -- see call_kv_quant for the write side,
    # which is the only place the engine pays for it.
    k_fp8, k_descale = _qfa_quant(cache_k.reshape(nb, BLOCK, NKV, D), D)
    v_fp8, v_descale = quant_v_by_sequence(cache_v.reshape(nb, BLOCK, NKV, D))
    q_fp8, q_descale = _qfa_quant(q, D)

    args = {
        "cu_seqlens_q": torch.tensor([0] + cum_q, dtype=torch.int32).npu(),
        "seqused_kv": torch.tensor(kv_lens, dtype=torch.int32).npu(),
        "mask_mode": 3,
        # Both bounds as _attach_qfa_inputs sets them: max_seqlen_q is the
        # cumulative last entry (the token count), and max_seqlen_kv is
        # max_model_len for every paged step, not this batch's kv. The plan
        # tiles on it -- AdjustSinnerAndSouter takes it -- so passing the
        # tighter real length here would measure a tiling the server never runs.
        "max_seqlen_q": cum_q[-1],
        "max_seqlen_kv": MAX_MODEL_LEN,
        "layout_q": "TND",
        "layout_q_descale": "TND",
        "layout_kv": "PA_BBND",
        "layout_out": "TND",
    }

    def call_metadata():
        return torch.ops._C_ascend.npu_quant_flash_attn_metadata(
            NQ, NKV, D, 1, v_descale=v_descale, **args)

    metadata = call_metadata()
    torch.npu.synchronize()

    def call_qfa():
        torch.ops._C_ascend.npu_quant_flash_attn(
            q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale, 1,
            block_table=table, attn_mask=mask, metadata=metadata,
            softmax_scale=D ** -0.5, **args)

    def call_fia():
        # The C8=0 baseline: _get_fia_params hands FIA the bf16 cache viewed as
        # (num_blocks, block_size, N*D) plus the block table, for chunked
        # prefill and decode alike.
        torch_npu.npu_fused_infer_attention_score(
            query=q, key=cache_k, value=cache_v, atten_mask=mask,
            block_table=table, input_layout="TND", block_size=BLOCK,
            actual_seq_lengths=cum_q, actual_seq_lengths_kv=kv_lens,
            num_key_value_heads=NKV, num_heads=NQ, scale=D ** -0.5, sparse_mode=3)

    def call_q_quant():
        # All _qfa_paged_call quantizes. K and V are already FP8 in the cache,
        # in the order PA_BBND reads, so the read side touches neither.
        _qfa_quant(q, D)

    # This step's new K/V, on their way into the cache. Prefill writes the whole
    # chunk, decode its 1 + 3 MTP tokens -- total_q either way.
    new_k = torch.randn(total_q, NKV, D, dtype=torch.bfloat16).npu()
    new_v_flat = torch.randn(total_q, NKV * D, dtype=torch.bfloat16).npu()
    # v_cache_scale_float_reciprocal is 1 / 2^(e8m0 - 127) cast to
    # model_config.dtype, so bf16 -- npu_quantize rejects a float32 scale
    # against a bf16 x with EZ1001.
    v_static_recip = (1.0 / torch.exp2(
        torch.randint(120, 135, (NKV * D,)).float() - 127)).to(torch.bfloat16).npu()

    def call_kv_quant():
        # Copied from AscendC8MXFPAttentionBackendImpl.forward, argument for
        # argument. K goes in 3-D with no scale_alg -- not the 2-D scale_alg=0
        # shape _qfa_quant uses for the query. V is flattened onto the
        # checkpoint's static per-channel reciprocal, which is why it is
        # npu_quantize and not a dynamic quant.
        torch_npu.npu_dynamic_mx_quant(new_k, dst_type=torch.float8_e4m3fn)
        torch_npu.npu_quantize(
            new_v_flat, v_static_recip, None, torch.float8_e4m3fn, -1, False)

    result = {
        "shape": label,
        "batch": batch,
        "q_len": q_len,
        "kv_len": kv_len,
        "c16_fia_ms": _time(call_fia, iters, warmup) * 1e3,
        "c8_qfa_ms": _time(call_qfa, iters, warmup) * 1e3,
        "q_quant_ms": _time(call_q_quant, iters, warmup) * 1e3,
        "kv_quant_ms": _time(call_kv_quant, iters, warmup) * 1e3,
        "metadata_ms": _time(call_metadata, iters, warmup) * 1e3,
    }
    print("BENCH-RESULT " + json.dumps(result), flush=True)
    return 0


def run_bench_sweep(args) -> int:
    """One subprocess per shape: the big ones can OOM, and losing the whole
    sweep to the last row would be a waste of a 20-minute setup."""
    import json

    rows = []
    for label, *_rest in bench_shapes():
        print(f"== {label}", flush=True)
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--shape", label,
             "--iters", str(args.iters), "--warmup", str(args.warmup), *shape_argv(args)],
            capture_output=True, text=True,
        )
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("BENCH-RESULT ")), None)
        if line:
            rows.append(json.loads(line[len("BENCH-RESULT "):]))
            last = rows[-1]
            print(f"   C16+FIA {last['c16_fia_ms']:.3f} ms   "
                  f"C8+QFA {last['c8_qfa_ms']:.3f} ms", flush=True)
        else:
            tail = (proc.stdout + proc.stderr).strip().splitlines()
            reason = tail[-1][:44] if tail else f"exit {proc.returncode}"
            rows.append({"shape": label, "error": reason})
            print(f"   FAILED: {reason}", flush=True)
            # The table column is 44 chars wide, which truncates every aclnn
            # error to the point of uselessness. Print the real tail here.
            for line in tail[-4:]:
                print(f"     | {line}", flush=True)
    _print_bench_table(rows, ATTN_LAYERS)
    return 0 if any("error" not in r for r in rows) else 1


def _c8_step_ms(row: dict, attn_layers: int | None) -> float:
    """What one attention layer costs the C8+QFA configuration in one step."""
    meta_share = row["metadata_ms"] / attn_layers if attn_layers else 0.0
    return row["c8_qfa_ms"] + row["q_quant_ms"] + row["kv_quant_ms"] + meta_share


def _print_bench_table(rows: list, attn_layers: int | None) -> None:
    ok = [r for r in rows if "error" not in r]

    head = (
        f"{'shape':<16}{'C16 FIA':>10}{'C8 QFA':>9}"
        f"{'q-quant':>9}{'kv-quant':>10}{'meta':>9}{'C8 step':>10}"
    )
    print("\n" + head)
    print("-" * len(head))
    for row in rows:
        if row.get("error"):
            print(f"{row['shape']:<16}{row['error']:>47}")
            continue
        print(
            f"{row['shape']:<16}{row['c16_fia_ms']:>10.3f}{row['c8_qfa_ms']:>9.3f}"
            f"{row['q_quant_ms']:>9.3f}{row['kv_quant_ms']:>10.3f}"
            f"{row['metadata_ms']:>9.3f}{_c8_step_ms(row, attn_layers):>10.3f}"
        )

    mark = "" if attn_layers else "*"
    head2 = f"{'shape':<16}{'op FIA/QFA':>14}{'config C16/C8' + mark:>17}"
    print("\n" + head2)
    print("-" * len(head2))
    for row in ok:
        op = row["c16_fia_ms"] / row["c8_qfa_ms"]
        cfg = row["c16_fia_ms"] / _c8_step_ms(row, attn_layers)
        print(f"{row['shape']:<16}{op:>13.2f}x{cfg:>16.2f}x")

    layers = f"/{attn_layers}" if attn_layers else ""
    print(
        "\nThe first table is one attention layer, one step, in milliseconds.\n"
        "  C16 FIA   FIA over the bf16 cache -- the C8=0 baseline.\n"
        "  C8 QFA    QuantFlashAttn over the MXFP8 cache, read in place.\n"
        "  q-quant   C8 only: this step's query, projected in bf16, made MXFP8.\n"
        "  kv-quant  C8 only: this step's new K/V, on their way into the cache.\n"
        "            The read side quantizes nothing -- the cache is already FP8.\n"
        "  meta      C8 only: the AICPU plan, once per step for ALL layers.\n"
        f"  C8 step   C8 QFA + q-quant + kv-quant + meta{layers}.\n"
        "\nThe second table is what those add up to.\n"
        "  op FIA/QFA     the operators alone, bf16 in against MXFP8 in. Unfair by\n"
        "                 construction and asked for anyway: it is what the kernel\n"
        "                 time becomes.\n"
        "  config C16/C8  C16 FIA / C8 step. The deployment answer.\n"
        "\nDecode is KV-bandwidth bound and MXFP8 is about half the bytes, so read\n"
        "the ratios as a curve against context length, not as single numbers."
    )
    if attn_layers:
        print(
            f"\nmeta is spread over {attn_layers} full-attention layers -- the stack is\n"
            "  hybrid 3:1 (full_attention_interval 4) and the linear layers never call\n"
            "  QFA. Override with --attn-layers if the served checkpoint differs."
        )
    else:
        print(
            "\n* C8 step and config C16/C8 EXCLUDE meta: --attn-layers 0 was passed, so\n"
            "  the plan has nothing to be spread over. Drop the flag for the model's own."
        )


def run_all_cases(args) -> int:
    """Every accuracy case, one subprocess each."""
    results = {}
    for case in CASES:
        print(f"== {case}")
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--case", case,
                               *shape_argv(args)],
                              capture_output=True, text=True)
        results[case] = proc.returncode == 0
        for line in (proc.stdout + proc.stderr).splitlines():
            if line.strip():
                print(f"   {line}")
        print(f"  [{'GREEN' if results[case] else 'RED'}] {case}")

    print()
    print("== summary ==")
    for case, ok in results.items():
        print(f"  {case}: {'GREEN' if ok else 'RED'}")
    print()
    if all(results.values()):
        print("Both operators agree once the quantization loss is taken out.")
    else:
        print("The two operators disagree beyond rounding -- a layout is wrong.")
    return 0 if all(results.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=CASES)
    ap.add_argument("--bench", action="store_true",
                    help="time C16+FIA against C8+QFA instead of comparing them")
    ap.add_argument("--all", action="store_true",
                    help="accuracy cases first, then the bench sweep")
    ap.add_argument("--shape", choices=BENCH_NAMES, help="bench a single shape")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)

    ap.add_argument("--model", choices=sorted(MODELS), default="27b", help="shape preset")
    for _name in SHAPE_FLAGS:
        ap.add_argument("--" + _name.replace("_", "-"), type=int, help="override the preset")
    args = ap.parse_args()
    apply_shape(args)
    if args.case:
        return run_case(args.case)
    if args.shape:
        row = next(r for r in bench_shapes() if r[0] == args.shape)
        return run_bench(*row, iters=args.iters, warmup=args.warmup)
    if args.bench:
        return run_bench_sweep(args)
    if args.all:
        # Accuracy first: it is the cheap half, and a layout error makes the
        # timings meaningless anyway. Both halves run whatever the first
        # returns, so one RED does not hide the other's numbers.
        print("=== accuracy ===", flush=True)
        accuracy = run_all_cases(args)
        print()
        print("=== performance ===", flush=True)
        performance = run_bench_sweep(args)
        print()
        print(f"accuracy: {'GREEN' if accuracy == 0 else 'RED'}   "
              f"performance: {'GREEN' if performance == 0 else 'RED'}")
        return accuracy or performance

    return run_all_cases(args)


if __name__ == "__main__":
    sys.exit(main())
