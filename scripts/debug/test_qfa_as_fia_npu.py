#!/usr/bin/env python3
"""Does the FIA-call-site swap actually run on device?

attention_v1 now calls QuantFlashAttn where it used to call FIA, quantizing
q/k/v on the spot with _qfa_quant. That function is mirrored below rather than
imported: importing attention_v1 on its own trips the device_op <-> fused_moe
circular import (the engine avoids it by registering the platform first). Keep
the two in step. Two things are worth checking before starting a server, both
about v_descale:

  * QFA groups V's scales down the sequence -- (T/64, N, D, 2) for TND,
    (Bn, Bs/64, N, D, 2) for PA_BBND -- while _qfa_quant groups along D like it
    does for q and k, giving (T, N, D/64, 2). Different layout, different size.
  * the paged branch hands over the cache as (num_blocks, block_size, N*D), so
    the last axis is not D at all and the scale reshape cannot even be formed.

So each shape is tried twice: once exactly as the swap does it today, once with
V quantized the way the doc specifies. Whichever pairs come back green say what
the call site has to look like.

Cases use the live 27B prefill/decode shapes (Nq=24 Nkv=4 D=256, block 128).
Each runs in its own subprocess: an AICPU abort poisons the device.

Usage (inside the serving container, no server running):
  python scripts/debug/test_qfa_as_fia_npu.py
  python scripts/debug/test_qfa_as_fia_npu.py --case dense-doc
"""

import argparse
import os
import subprocess
import sys

NQ, NKV, D, BLOCK, WINDOW = 24, 4, 256, 128, 64
PREFILL_LEN = 1594
DECODE_REQS, DECODE_KV = 4, 300
CASES = ["dense-asis", "dense-doc", "paged-asis", "paged-doc"]


def quant_v_by_sequence(value, seq_lens=None):
    """V quantized down the sequence, the layout QFA documents."""
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

    paged = name.startswith("paged")
    as_is = name.endswith("asis")
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()

    if paged:
        nb = 64
        q = torch.randn(DECODE_REQS, NQ, D, dtype=torch.bfloat16).npu()
        cache_k = torch.randn(nb, BLOCK, NKV, D, dtype=torch.bfloat16).npu()
        cache_v = torch.randn(nb, BLOCK, NKV, D, dtype=torch.bfloat16).npu()
        blocks_per_req = (DECODE_KV + BLOCK - 1) // BLOCK
        table = torch.arange(1, DECODE_REQS * blocks_per_req + 1, dtype=torch.int32)
        table = table.reshape(DECODE_REQS, blocks_per_req).npu()
        q_lens, kv_lens = [1] * DECODE_REQS, [DECODE_KV] * DECODE_REQS
        # as-is: the call site receives the cache as (num_blocks, block_size, N*D)
        key = cache_k.reshape(nb, BLOCK, NKV * D) if as_is else cache_k
        value = cache_v.reshape(nb, BLOCK, NKV * D) if as_is else cache_v
        layout_kv = "PA_BBND"
    else:
        q = torch.randn(PREFILL_LEN, NQ, D, dtype=torch.bfloat16).npu()
        key = torch.randn(PREFILL_LEN, NKV, D, dtype=torch.bfloat16).npu()
        value = torch.randn(PREFILL_LEN, NKV, D, dtype=torch.bfloat16).npu()
        table = None
        q_lens, kv_lens = [PREFILL_LEN], [PREFILL_LEN]
        layout_kv = "TND"

    q_fp8, q_descale = _qfa_quant(q, D)
    k_fp8, k_descale = _qfa_quant(key, D)
    if as_is:
        v_fp8, v_descale = _qfa_quant(value, D)
    else:
        v_fp8, v_descale = quant_v_by_sequence(value, None if paged else kv_lens)
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
        "seqused_kv": torch.tensor(kv_lens, dtype=torch.int32).npu(),
        "mask_mode": 3,
        "max_seqlen_q": max(q_lens),
        "max_seqlen_kv": max(kv_lens),
        "layout_q": "TND",
        "layout_q_descale": "TND",
        "layout_kv": layout_kv,
        "layout_out": "TND",
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
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=CASES)
    args = ap.parse_args()
    if args.case:
        return run_case(args.case)

    results = {}
    for case in CASES:
        print(f"== {case} " + ("(exactly what attention_v1 does now)" if case.endswith("asis")
                               else "(V grouped down the sequence, per the doc)"))
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--case", case],
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
    print("== reading ==")
    if results["dense-doc"] and not results["dense-asis"]:
        print("  Prefill needs V grouped down the sequence; quantizing it like q/k is not enough.")
    if results["paged-doc"] and not results["paged-asis"]:
        print("  Decode needs the cache as (Bn, Bs, N, D) and the same V treatment.")
    if all(results.values()):
        print("  Everything runs as written -- the swap needs no further shaping.")
    return 0 if results["dense-doc"] and results["paged-doc"] else 1


if __name__ == "__main__":
    sys.exit(main())
