#!/usr/bin/env python3
"""Turn a QFA dump into a precision report, splitting quantization from compute.

Reads the pairs written by vllm_ascend/attention/qfa_dump.py and answers three
questions separately, because "QFA is N% off" on its own does not say whose
fault it is:

  quantization loss  bf16 K/V vs the same tensors dequantized back out of the
                     cache. This is what MXFP8 storage costs, before attention
                     runs at all.
  compute loss       QFA's output vs a float reference fed the *dequantized*
                     K/V -- identical inputs, so the only difference left is
                     doing the matmuls in FP8.
  end-to-end loss    QFA's output vs a float reference fed the original bf16
                     K/V. Should land near the quadrature sum of the two above;
                     if it does not, one of the other two is measuring wrong.

It also reports what a per-sequence optimal V scale would have achieved, which
is the gap the single-operator benchmark cannot see: that one picks the optimum,
while serving uses the checkpoint's static per-channel scale.

Runs anywhere PyTorch is installed -- CPU is fine, no NPU and no torch_npu.
Pass --selftest to check the dequantization against synthetic data with a known
answer before trusting any real numbers.

Usage:
  python3 analyze_qfa_dump.py DUMP_DIR
  python3 analyze_qfa_dump.py --selftest
Final line is [GREEN] (exit 0) or [RED] (exit 1).
"""

import argparse
import math
import sys
from pathlib import Path

import torch

# Mirrors vllm_ascend/device/mxfp_kv_cache.py. K scales group along head_dim,
# V scales along the sequence; both store 2 values per 64-wide slot, i.e. one
# E8M0 exponent per 32 elements.
SCALE_GROUP_SIZE = 64
VALUES_PER_GROUP = 2
QUANT_GROUP_SIZE = 32
E8M0_BIAS = 127


def e8m0_to_float(raw: torch.Tensor) -> torch.Tensor:
    """E8M0 byte -> multiplier. 0 means 'no scale' and is read as the neutral 1.0.

    mxfp_c8.py's process_weights_after_loading() does the same rewrite: a minmax
    calibrator emits 0 for an all-zero channel, and 2^-127 there would send any
    non-zero activation to inf on the reciprocal.
    """
    raw = raw.to(torch.int64)
    exponent = torch.where(raw == 0, torch.full_like(raw, E8M0_BIAS), raw) - E8M0_BIAS
    return torch.pow(torch.tensor(2.0, dtype=torch.float64), exponent.to(torch.float64))


def dequant_along_last(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 + E8M0 grouped along the last axis -> float64, same leading shape.

    Used for q, whose scale is [..., D//64, 2] per token: one exponent per
    QUANT_GROUP_SIZE consecutive head_dim elements, the same grouping K uses.
    """
    lead = values.shape[:-1]
    d = values.shape[-1]
    scales = e8m0_to_float(scale).reshape(*lead, d // QUANT_GROUP_SIZE)
    scales = scales.repeat_interleave(QUANT_GROUP_SIZE, dim=-1)
    return values.to(torch.float64) * scales


def dequant_k_cache(key_cache: torch.Tensor, key_scale_cache: torch.Tensor) -> torch.Tensor:
    """[nb, bs, N, D] FP8 + [nb, bs, N, D//64, 2] E8M0 -> float64 [nb, bs, N, D]."""
    nb, bs, n, d = key_cache.shape
    values = key_cache.to(torch.float64)
    scales = e8m0_to_float(key_scale_cache)  # [nb, bs, N, D//64, 2]
    # Each stored scale covers QUANT_GROUP_SIZE consecutive head_dim elements.
    scales = scales.reshape(nb, bs, n, d // QUANT_GROUP_SIZE)
    scales = scales.repeat_interleave(QUANT_GROUP_SIZE, dim=-1)
    return values * scales


def dequant_v_cache(value_cache: torch.Tensor, value_scale_cache: torch.Tensor) -> torch.Tensor:
    """[nb, bs, N, D] FP8 + [nb, bs//64, N, D, 2] E8M0 -> float64 [nb, bs, N, D].

    The V scale groups along the sequence axis, so a stored slot covers
    QUANT_GROUP_SIZE consecutive tokens of one (head, dim) channel.
    """
    nb, bs, n, d = value_cache.shape
    values = value_cache.to(torch.float64)
    scales = e8m0_to_float(value_scale_cache)  # [nb, bs//64, N, D, 2]
    # -> [nb, bs//32, N, D] with the group axis adjacent to the token axis.
    scales = scales.permute(0, 1, 4, 2, 3).reshape(nb, bs // QUANT_GROUP_SIZE, n, d)
    scales = scales.repeat_interleave(QUANT_GROUP_SIZE, dim=1)
    return values * scales


def gather_kv(cache: torch.Tensor, block_table_row: torch.Tensor, kv_len: int) -> torch.Tensor:
    """Flatten paged [nb, bs, N, D] into contiguous [kv_len, N, D] for one sequence."""
    block_size = cache.shape[1]
    needed = (kv_len + block_size - 1) // block_size
    blocks = [cache[int(block_table_row[i])] for i in range(needed)]
    return torch.cat(blocks, dim=0)[:kv_len]


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
    causal: bool = True,
) -> torch.Tensor:
    """Exact attention in float64. query [Tq, H, D], key/value [Tkv, N, D]."""
    tq, h, _ = query.shape
    tkv, n, _ = key.shape
    q = query.to(torch.float64).permute(1, 0, 2)  # [H, Tq, D]
    k = key.to(torch.float64).permute(1, 0, 2)  # [N, Tkv, D]
    v = value.to(torch.float64).permute(1, 0, 2)

    # GQA: every group of H/N query heads shares one KV head.
    repeat = h // n
    k = k.repeat_interleave(repeat, dim=0)
    v = v.repeat_interleave(repeat, dim=0)

    scores = torch.matmul(q, k.transpose(-1, -2)) * softmax_scale  # [H, Tq, Tkv]
    if causal and tq > 1:
        # Query token i corresponds to KV position tkv - tq + i.
        offset = tkv - tq
        rows = torch.arange(tq).unsqueeze(1)
        cols = torch.arange(tkv).unsqueeze(0)
        scores = scores.masked_fill(cols > rows + offset, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v).permute(1, 0, 2)  # [Tq, H, D]


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Relative L2 error, accumulated in float64 (float32 drifts on ~1e7 elements)."""
    a = actual.to(torch.float64)
    e = expected.to(torch.float64)
    denom = torch.linalg.vector_norm(e)
    if denom == 0:
        return float("nan")
    return float(torch.linalg.vector_norm(a - e) / denom)


def cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.to(torch.float64).flatten()
    e = expected.to(torch.float64).flatten()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(e)
    if denom == 0:
        return float("nan")
    return float(torch.dot(a, e) / denom)


def optimal_v_scale_error(value_bf16: torch.Tensor) -> float:
    """Error a per-sequence optimal E8M0 V scale would leave, for comparison.

    The single-operator benchmark quantizes V with a scale derived from the data
    it is about to quantize; serving uses whatever the checkpoint calibrated.
    The gap between this number and the measured V error is what that costs.
    """
    v = value_bf16.to(torch.float64)
    fp8_max = 448.0  # e4m3 max magnitude
    amax = v.abs().amax(dim=0, keepdim=True).clamp(min=1e-30)
    exponent = torch.floor(torch.log2(fp8_max / amax))
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float64), exponent)
    quantized = (v * scale).to(torch.float8_e4m3fn).to(torch.float64) / scale
    return rel_l2(quantized, v)


def analyze_pair(cache_write: dict, qfa_call: dict, label: str) -> dict:
    """Report one layer/call: quantization, compute and end-to-end losses."""
    print(f"\n=== {label} ===")

    key_bf16 = cache_write["key_bf16"]
    value_bf16 = cache_write["value_bf16"]
    num_new = int(cache_write["num_actual_tokens"])

    op_kwargs = qfa_call["op_kwargs"]
    seqused_kv = op_kwargs["seqused_kv"].to(torch.int64)
    block_table = qfa_call["block_table_local"]
    query_bf16 = qfa_call["query_bf16"]
    attn_output = qfa_call["attn_output"]
    softmax_scale = float(qfa_call["softmax_scale"])
    num_heads = int(qfa_call["num_heads"])
    head_size = int(qfa_call["head_size"])

    key_deq = dequant_k_cache(qfa_call["key_cache"], qfa_call["key_scale_cache"])
    value_deq = dequant_v_cache(qfa_call["value_cache"], qfa_call["value_scale_cache"])

    if seqused_kv.numel() != 1:
        print(f"  batch of {seqused_kv.numel()} sequences; this report handles batch 1, skipping")
        return {}
    kv_len = int(seqused_kv[0])
    print(f"  kv_len={kv_len} new_tokens={num_new} q_tokens={query_bf16.shape[0]} heads={num_heads} D={head_size}")

    k_cache_seq = gather_kv(key_deq, block_table[0], kv_len)
    v_cache_seq = gather_kv(value_deq, block_table[0], kv_len)

    results: dict[str, float] = {}

    # --- 1. quantization loss, measurable only against this step's bf16 -----
    if kv_len == num_new:
        # Whole sequence was written this step (a prefill): bf16 covers all of it.
        results["k_quant"] = rel_l2(k_cache_seq, key_bf16)
        results["v_quant"] = rel_l2(v_cache_seq, value_bf16)
        results["v_quant_optimal"] = optimal_v_scale_error(value_bf16)
        print(f"  [quantization] K  rel_l2 = {results['k_quant'] * 100:.3f}%   (dynamic per-32 E8M0)")
        print(f"  [quantization] V  rel_l2 = {results['v_quant'] * 100:.3f}%   (checkpoint static per-channel)")
        opt = results["v_quant_optimal"] * 100
        print(f"  [quantization] V  rel_l2 = {opt:.3f}%   if scale were per-sequence optimal")
        gap = results["v_quant"] - results["v_quant_optimal"]
        print(f"                 -> static scale costs {gap * 100:+.3f} percentage points on V")
    else:
        tail_k = k_cache_seq[kv_len - num_new :]
        tail_v = v_cache_seq[kv_len - num_new :]
        results["k_quant"] = rel_l2(tail_k, key_bf16)
        results["v_quant"] = rel_l2(tail_v, value_bf16)
        print(f"  [quantization] measured on this step's {num_new} new token(s) only:")
        kq, vq = results["k_quant"] * 100, results["v_quant"] * 100
        print(f"                 K rel_l2 = {kq:.3f}%  V rel_l2 = {vq:.3f}%")
        print("                 (cache holds history whose bf16 truth was never dumped)")

    # --- 2. compute loss: same dequantized inputs, float reference ----------
    # The reference has to be fed the *quantized* q, not query_bf16: QFA never
    # sees the bf16 query, and q's quantization error is amplified through
    # softmax's exp, so mixing it in here would be charged to the operator.
    q_deq = dequant_along_last(qfa_call["q_fp8"], qfa_call["q_descale"])
    results["q_quant"] = rel_l2(q_deq, query_bf16)
    print(f"  [quantization] q  rel_l2 = {results['q_quant'] * 100:.3f}%   (dynamic per-32 E8M0, recomputed each step)")

    ref_from_cache = reference_attention(q_deq, k_cache_seq, v_cache_seq, softmax_scale)
    out = attn_output.reshape(-1, num_heads, head_size)
    results["compute"] = rel_l2(out, ref_from_cache)
    results["compute_cos"] = cosine(out, ref_from_cache)
    print("  [compute]      QFA vs float64 on identical (dequantized q/K/V):")
    print(f"                 rel_l2 = {results['compute'] * 100:.3f}%   cos = {results['compute_cos']:.6f}")

    # --- 3. end-to-end, only meaningful when bf16 covers the whole sequence -
    if kv_len == num_new:
        ref_from_bf16 = reference_attention(query_bf16, key_bf16, value_bf16, softmax_scale)
        results["end_to_end"] = rel_l2(out, ref_from_bf16)
        print("  [end-to-end]   QFA vs float64 on the original bf16 K/V:")
        print(f"                 rel_l2 = {results['end_to_end'] * 100:.3f}%")
        implied = math.sqrt(max(results["end_to_end"] ** 2 - results["compute"] ** 2, 0.0))
        comp = results["compute"] * 100
        print(f"                 -> quantization component {implied * 100:.3f}%, compute component {comp:.3f}%")
    else:
        print("  [end-to-end]   skipped: needs a step where bf16 covers the whole sequence (a prefill)")

    return results


def selftest() -> int:
    """Check dequantization round-trips before any real dump is trusted."""
    print("=== selftest: dequantization round-trip ===")
    torch.manual_seed(0)
    nb, bs, n, d = 2, 128, 2, 256
    ok = True

    # K: quantize per 32 head_dim elements with an exact power-of-two scale, so
    # a correct dequantizer reproduces the input bit-for-bit.
    k = torch.randn(nb, bs, n, d, dtype=torch.float64)
    groups = d // QUANT_GROUP_SIZE
    kg = k.reshape(nb, bs, n, groups, QUANT_GROUP_SIZE)
    amax = kg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    exponent = torch.floor(torch.log2(448.0 / amax))
    kq = (kg * torch.pow(torch.tensor(2.0, dtype=torch.float64), exponent)).to(torch.float8_e4m3fn)
    k_cache = kq.reshape(nb, bs, n, d)
    k_scale = (E8M0_BIAS - exponent.squeeze(-1)).to(torch.uint8).reshape(nb, bs, n, groups // 2, 2)
    k_deq = dequant_k_cache(k_cache, k_scale)
    expected = kq.to(torch.float64).reshape(nb, bs, n, d) * torch.pow(
        torch.tensor(2.0, dtype=torch.float64), -exponent
    ).squeeze(-1).repeat_interleave(QUANT_GROUP_SIZE, dim=-1)
    k_err = rel_l2(k_deq, expected)
    print(f"  K dequant round-trip rel_l2 = {k_err:.3e}  (expect 0)")
    ok &= k_err < 1e-12

    # V: one constant E8M0 exponent everywhere, matching the static broadcast
    # that value_scale_cache.copy_() performs in reshape_and_cache.
    v_exp = 120
    v = torch.randn(nb, bs, n, d, dtype=torch.float64)
    v_mult = 2.0 ** (v_exp - E8M0_BIAS)
    vq = (v / v_mult).to(torch.float8_e4m3fn)
    v_scale = torch.full((nb, bs // SCALE_GROUP_SIZE, n, d, VALUES_PER_GROUP), v_exp, dtype=torch.uint8)
    v_deq = dequant_v_cache(vq.reshape(nb, bs, n, d), v_scale)
    v_err = rel_l2(v_deq, vq.to(torch.float64) * v_mult)
    print(f"  V dequant round-trip rel_l2 = {v_err:.3e}  (expect 0)")
    ok &= v_err < 1e-12

    # Reference attention against a hand-rolled single-head case.
    q1 = torch.randn(4, 2, 8, dtype=torch.float64)
    k1 = torch.randn(4, 2, 8, dtype=torch.float64)
    v1 = torch.randn(4, 2, 8, dtype=torch.float64)
    got = reference_attention(q1, k1, v1, 1.0, causal=True)
    manual = torch.empty_like(got)
    for h in range(2):
        s = (q1[:, h] @ k1[:, h].T).masked_fill(
            torch.arange(4).unsqueeze(0) > torch.arange(4).unsqueeze(1), float("-inf")
        )
        manual[:, h] = torch.softmax(s, dim=-1) @ v1[:, h]
    attn_err = rel_l2(got, manual)
    print(f"  causal attention vs manual  rel_l2 = {attn_err:.3e}  (expect 0)")
    ok &= attn_err < 1e-12

    print()
    if ok:
        print("[GREEN]")
        return 0
    print("[RED]")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump_dir", nargs="?", help="directory written by VLLM_ASCEND_QFA_DUMP_DIR")
    parser.add_argument("--selftest", action="store_true", help="verify the dequantizers, no dump needed")
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if not args.dump_dir:
        parser.error("give a dump directory, or --selftest")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vllm-ascend" / "feat-qfa-dump"))
    try:
        from vllm_ascend.attention.qfa_dump import load_dump
    except ImportError:
        # Standalone fallback: the same restore logic, without importing vllm_ascend.
        def load_dump(path):
            def restore(obj):
                if isinstance(obj, dict):
                    if "__fp8_dtype__" in obj:
                        name = obj["__fp8_dtype__"].removeprefix("torch.")
                        dtype = getattr(torch, name, None)
                        return obj["bytes"] if dtype is None else obj["bytes"].view(dtype)
                    return {k: restore(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return type(obj)(restore(v) for v in obj)
                return obj

            return restore(torch.load(path, map_location="cpu", weights_only=False))

    dump_dir = Path(args.dump_dir)
    if not dump_dir.is_dir():
        print(f"[RED] not a directory: {dump_dir}")
        return 1

    pairs = []
    for call_file in sorted(dump_dir.glob("*__qfa_call.pt")):
        write_file = Path(str(call_file).replace("__qfa_call.pt", "__cache_write.pt"))
        if write_file.is_file():
            pairs.append((write_file, call_file))
        else:
            print(f"  (skipping {call_file.name}: no matching cache_write)")

    if not pairs:
        print(f"[RED] no complete dump pairs under {dump_dir}")
        return 1

    print(f"found {len(pairs)} dump pair(s) under {dump_dir}")
    all_results = []
    for write_file, call_file in pairs:
        label = call_file.name.replace("__qfa_call.pt", "")
        results = analyze_pair(load_dump(str(write_file)), load_dump(str(call_file)), label)
        if results:
            all_results.append(results)

    if not all_results:
        print("\n[RED] nothing analyzable (batch>1 dumps only?)")
        return 1

    print("\n=== summary across layers ===")
    for key, title in (
        ("q_quant", "q quantization"),
        ("k_quant", "K quantization"),
        ("v_quant", "V quantization (static scale)"),
        ("v_quant_optimal", "V quantization (optimal scale)"),
        ("compute", "QFA compute"),
        ("end_to_end", "end-to-end"),
    ):
        values = [r[key] for r in all_results if key in r]
        if values:
            print(f"  {title:<34} mean {sum(values) / len(values) * 100:6.3f}%   max {max(values) * 100:6.3f}%")

    print("\n[GREEN]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
