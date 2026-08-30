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

Cases use the live 27B prefill/decode shapes (Nq=24 Nkv=4 D=256, block 128).
Each runs in its own subprocess: an AICPU abort poisons the device.

Usage (inside the serving container, no server running):
  python scripts/debug/test_qfa_as_fia_npu.py
  python scripts/debug/test_qfa_as_fia_npu.py --case dense
"""

import argparse
import os
import subprocess
import sys

NQ, NKV, D, BLOCK, WINDOW = 24, 4, 256, 128, 64
PREFILL_LEN = 1594
DECODE_REQS, DECODE_KV = 4, 300
CASES = ["dense", "paged"]


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
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=CASES)
    args = ap.parse_args()
    if args.case:
        return run_case(args.case)

    results = {}
    for case in CASES:
        print(f"== {case}")
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
    if all(results.values()):
        print("Both paths run; the swap is shaped the way the operator wants.")
    else:
        print("Read the failing case above before starting a server.")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
