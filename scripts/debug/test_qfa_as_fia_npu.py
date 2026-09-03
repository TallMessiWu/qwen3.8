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
  paged  everything else: the cache arrives as (num_blocks, block_size, N*D),
         so the heads get split back out for PA_BBND, and seqused_kv is
         per-sequence here.

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

--bench times the two operators instead of comparing them, over the shapes the
27B config actually produces. What it answers is "what will QFA be worth once
the cache is MXFP8", which is measurable today: QFA reads MXFP8 whatever wrote
it, so its main-op time is already the post-migration number. The on-the-fly
quantization is timed separately rather than subtracted -- it is the cost that
disappears, not part of the operator.

Attention decode is bound by KV bandwidth and QFA reads half the bytes, so the
interesting output is how the ratio moves with context length, not any single
number.

Usage (inside the serving container, no server running):
  python scripts/debug/test_qfa_as_fia_npu.py
  python scripts/debug/test_qfa_as_fia_npu.py --case dense
  python scripts/debug/test_qfa_as_fia_npu.py --bench
  python scripts/debug/test_qfa_as_fia_npu.py --model 35b
  python scripts/debug/test_qfa_as_fia_npu.py --model 35b --bench
  python scripts/debug/test_qfa_as_fia_npu.py --model 35b --all   # both halves
  python scripts/debug/test_qfa_as_fia_npu.py --bench --shape decode-b32-16k
"""

import argparse
import os
import subprocess
import sys

# Shape presets. Both served models are D=256, but the head counts and the
# kernel block size differ -- and the decode bandwidth ratio this script exists
# to measure depends on exactly those (NKV * D bytes per token, BLOCK per page),
# so measuring 27B's numbers against a 35B server would answer nothing.
MODELS = {
    "27b": {"num_heads": 24, "num_kv_heads": 4, "head_dim": 256,
            "block_size": 128, "prefill_len": 1594},
    "35b": {"num_heads": 16, "num_kv_heads": 2, "head_dim": 256,
            "block_size": 512, "prefill_len": 1552},
}

# Defaults are 27B's, as they always were. apply_shape() overwrites them from
# --model / the per-field flags before anything reads them; the module-level
# names stay so the call sites below need no threading.
NQ, NKV, D, BLOCK, WINDOW = 24, 4, 256, 128, 64
PREFILL_LEN = 1594
DECODE_REQS, DECODE_KV = 4, 300

# WINDOW is MXFP8's scale grouping, not a model shape -- it stays 64.
SHAPE_FLAGS = ("num_heads", "num_kv_heads", "head_dim", "block_size", "prefill_len")


def apply_shape(args) -> None:
    """Resolve the preset plus any per-field override into the globals."""
    global NQ, NKV, D, BLOCK, PREFILL_LEN
    preset = MODELS[args.model]
    resolved = {name: getattr(args, name) or preset[name] for name in SHAPE_FLAGS}
    NQ = resolved["num_heads"]
    NKV = resolved["num_kv_heads"]
    D = resolved["head_dim"]
    BLOCK = resolved["block_size"]
    PREFILL_LEN = resolved["prefill_len"]
    print(
        f"shapes: model={args.model} num_heads={NQ} num_kv_heads={NKV} "
        f"head_dim={D} block_size={BLOCK} prefill_len={PREFILL_LEN}",
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

# The shapes the live 27B config produces. Prefill is one request at a time with
# q == kv (PrefillNoCache, TND); 16384 is max_num_batched_tokens, the largest
# single chunk, and 1594 is the multimodal prompt that first exercised this.
# Decode carries 1 + 3 MTP tokens per request, up to max_num_seqs=32, with kv
# running out to max_model_len=133120.
BENCH_SHAPES = [
    ("prefill-512", "dense", 1, 512, 512),
    ("prefill-1594", "dense", 1, PREFILL_LEN, PREFILL_LEN),
    ("prefill-4k", "dense", 1, 4096, 4096),
    ("prefill-16k", "dense", 1, 16384, 16384),
    ("decode-b32-1k", "paged", 32, 4, 1024),
    ("decode-b32-4k", "paged", 32, 4, 4096),
    ("decode-b32-16k", "paged", 32, 4, 16384),
    ("decode-b8-32k", "paged", 8, 4, 32768),
    ("decode-b1-128k", "paged", 1, 4, 131072),
]
BENCH_NAMES = [row[0] for row in BENCH_SHAPES]


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
    from test_junlin_qfa_npu import bootstrap_ops

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


def run_bench(label: str, kind: str, batch: int, q_len: int, kv_len: int, iters: int, warmup: int) -> int:
    """Time FIA, QFA, the throwaway quantization, and the per-step metadata."""
    import json

    import torch
    import torch_npu  # noqa: F401

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_junlin_qfa_npu import bootstrap_ops

    torch.npu.set_device(int(os.environ.get("QFA_DEVICE", "0")))
    bootstrap_ops()

    def _qfa_quant(x, d):
        """Mirror of attention_v1._qfa_quant (see run_case)."""
        fp8, scale = torch_npu.npu_dynamic_mx_quant(
            x.reshape(-1, d), dst_type=torch.float8_e4m3fn, scale_alg=0)
        return (
            fp8.reshape(x.shape),
            scale.view(torch.uint8).reshape(*x.shape[:-1], d // 64, 2).view(torch.float8_e8m0fnu),
        )

    paged = kind == "paged"
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()
    total_q = batch * q_len
    q = torch.randn(total_q, NQ, D, dtype=torch.bfloat16).npu()
    cum_q = [(i + 1) * q_len for i in range(batch)]
    kv_lens = [kv_len] * batch

    if paged:
        blocks_per_req = (kv_len + BLOCK - 1) // BLOCK
        nb = batch * blocks_per_req
        cache_k = torch.randn(nb, BLOCK, NKV * D, dtype=torch.bfloat16).npu()
        cache_v = torch.randn(nb, BLOCK, NKV * D, dtype=torch.bfloat16).npu()
        table = torch.arange(nb, dtype=torch.int32).reshape(batch, blocks_per_req).npu()
        fia_key, fia_value = cache_k, cache_v
        k_src = cache_k.reshape(nb, BLOCK, NKV, D)
        v_src = cache_v.reshape(nb, BLOCK, NKV, D)
        quant_kv = lambda: (_qfa_quant(k_src, D), quant_v_by_sequence(v_src))  # noqa: E731
        kv_args = {"seqused_kv": torch.tensor(kv_lens, dtype=torch.int32).npu()}
        layout_kv = "PA_BBND"
        fia_kvlen = kv_lens
    else:
        key = torch.randn(total_q, NKV, D, dtype=torch.bfloat16).npu()
        value = torch.randn(total_q, NKV, D, dtype=torch.bfloat16).npu()
        table = None
        fia_key, fia_value = key, value
        quant_kv = lambda: (_qfa_quant(key, D), quant_v_by_sequence(value, kv_lens))  # noqa: E731
        cum_kv = [(i + 1) * kv_len for i in range(batch)]
        kv_args = {"cu_seqlens_kv": torch.tensor([0] + cum_kv, dtype=torch.int32).npu()}
        layout_kv = "TND"
        fia_kvlen = cum_kv

    (k_fp8, k_descale), (v_fp8, v_descale) = quant_kv()
    q_fp8, q_descale = _qfa_quant(q, D)
    args = {
        "cu_seqlens_q": torch.tensor([0] + cum_q, dtype=torch.int32).npu(),
        "mask_mode": 3,
        "max_seqlen_q": max(cum_q),
        "max_seqlen_kv": kv_len,
        "layout_q": "TND",
        "layout_q_descale": "TND",
        "layout_kv": layout_kv,
        "layout_out": "TND",
        **kv_args,
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
        torch_npu.npu_fused_infer_attention_score(
            query=q, key=fia_key, value=fia_value, atten_mask=mask,
            block_table=table, input_layout="TND", block_size=BLOCK,
            actual_seq_lengths=cum_q, actual_seq_lengths_kv=fia_kvlen,
            num_key_value_heads=NKV, num_heads=NQ, scale=D ** -0.5, sparse_mode=3)

    def call_quant():
        # q only. K/V come out of the C8 cache already MXFP8 -- quantizing them
        # per step was the pre-C8 path and is not what the engine does now.
        _qfa_quant(q, D)

    result = {
        "shape": label,
        "kind": kind,
        "batch": batch,
        "q_len": q_len,
        "kv_len": kv_len,
        "fia_ms": _time(call_fia, iters, warmup) * 1e3,
        "qfa_ms": _time(call_qfa, iters, warmup) * 1e3,
        "quant_ms": _time(call_quant, iters, warmup) * 1e3,
        "metadata_ms": _time(call_metadata, iters, warmup) * 1e3,
    }
    print("BENCH-RESULT " + json.dumps(result), flush=True)
    return 0


def run_bench_sweep(args) -> int:
    """One subprocess per shape: the big ones can OOM, and losing the whole
    sweep to the last row would be a waste of a 20-minute setup."""
    import json

    rows = []
    for label, *_rest in BENCH_SHAPES:
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
            print(f"   FIA {last['fia_ms']:.3f} ms   QFA {last['qfa_ms']:.3f} ms", flush=True)
        else:
            tail = (proc.stdout + proc.stderr).strip().splitlines()
            reason = tail[-1][:44] if tail else f"exit {proc.returncode}"
            rows.append({"shape": label, "error": reason})
            print(f"   FAILED: {reason}", flush=True)
            # The table column is 44 chars wide, which truncates every aclnn
            # error to the point of uselessness. Print the real tail here.
            for line in tail[-4:]:
                print(f"     | {line}", flush=True)
    _print_bench_table(rows)
    return 0 if any("error" not in r for r in rows) else 1


def _print_bench_table(rows: list) -> None:
    head = (
        f"{'shape':<16}{'FIA ms':>9}{'QFA ms':>9}{'speedup':>9}"
        f"{'q-quant':>10}{'step ms':>10}{'meta ms':>9}"
    )
    print("\n" + head)
    print("-" * len(head))
    for row in rows:
        if row.get("error"):
            print(f"{row['shape']:<16}{row['error']:>46}")
            continue
        step = row["qfa_ms"] + row["quant_ms"]
        speedup = row["fia_ms"] / row["qfa_ms"] if row["qfa_ms"] else float("nan")
        print(
            f"{row['shape']:<16}{row['fia_ms']:>9.3f}{row['qfa_ms']:>9.3f}"
            f"{speedup:>8.2f}x{row['quant_ms']:>10.3f}{step:>10.3f}{row['metadata_ms']:>9.3f}"
        )
    print(
        "\nspeedup = FIA / QFA, per attention layer, both reading what the C8 cache holds:\n"
        "  FIA bf16, QFA MXFP8. Decode is KV-bandwidth bound and QFA reads half the\n"
        "  bytes, so watch how the ratio moves with context length.\n"
        "q-quant = quantizing this step's query, per layer per step. K/V are\n"
        "  not quantized: the C8 cache already holds them as MXFP8. Not part of\n"
        "  the operator; never subtract it.\n"
        "step ms = QFA + q-quant, one attention layer per decode step.\n"
        "meta ms = the AICPU plan, paid once per step for all layers, not per layer.\n"
        "  It is not optional: the doc calls it a scheduling hint, but passing None\n"
        "  is rejected outright (EZ0004) -- hence no no-metadata column."
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
    ap.add_argument("--bench", action="store_true", help="time the two operators instead of comparing")
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
        row = next(r for r in BENCH_SHAPES if r[0] == args.shape)
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
