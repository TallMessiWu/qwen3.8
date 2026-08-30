#!/usr/bin/env python3
"""Pin down why the QFA metadata op aborts on a real 27B prefill.

A 1594-token prefill kills the AICPU metadata kernel (errcode 0x2a,
kernelName=QuantFlashAttnMetadata) while the single-op smoke script stays green.
Two explanations fit that evidence, and they need different fixes:

  A  the split-plan buffer is too small. The binding allocates a fixed 4096
     int32, but the plan grows with the work being scheduled, and the operator
     reference sizes it as ((36 + 72) * batch * head_num + 1) * 16 rounded up to
     4096 -- 45056 for this shape. If so, the abort appears once the sequence is
     long enough and is independent of the block table.

  B  the block table's zero padding. vLLM hands over a table sized for
     max_model_len (1040 columns here) with unused entries left at 0, while the
     op doc says block_table values may only be positive integers. If so, the
     abort tracks the padded table, not the sequence length.

So: sweep the sequence length with a tight table (A's signal), then repeat the
longest one with a vLLM-shaped padded table (B's signal).

Each combination runs in its own subprocess -- an AICPU abort poisons the device
for everything that follows, so a shared process could only ever report the first
failure.

No weights, no golden comparison: this only asks whether the launch survives.

Usage (inside the serving container):
  python scripts/debug/diag_qfa_metadata_size.py
  python scripts/debug/diag_qfa_metadata_size.py --len 1594 --cols 1040  # one combo
"""

import argparse
import os
import subprocess
import sys

NQ, NKV, D = 24, 4, 256  # Qwen3.8-27B per-rank head shape
BLOCK = 128
WINDOW = 64
SWEEP_LENS = [256, 512, 1024, 1594]
VLLM_TABLE_COLS = 1040  # ceil(max_model_len 133120 / 128)


def run_one(seq_len: int, table_cols: int) -> int:
    import torch
    import torch_npu  # noqa: F401

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_junlin_qfa_npu import bootstrap_ops, causal_mask_npu

    torch.npu.set_device(int(os.environ.get("QFA_DEVICE", "0")))
    bootstrap_ops()

    fp8, e8m0 = torch.float8_e4m3fn, torch.float8_e8m0fnu
    blocks = (seq_len + BLOCK - 1) // BLOCK
    cols = table_cols or blocks
    # Block ids start at 1 so every used entry is a positive integer; the cache
    # holds one extra block so id `blocks` is still in range.
    table = torch.zeros(1, cols, dtype=torch.int32)
    table[0, :blocks] = torch.arange(1, blocks + 1, dtype=torch.int32)
    nb = blocks + 1

    def scales(shape):  # e8m0 byte 127 == 2^0, a valid finite scale
        return torch.full(shape, 127, dtype=torch.uint8).npu().view(e8m0)

    q = torch.randn(seq_len, NQ, D).npu().to(fp8)
    k = torch.randn(nb, BLOCK, NKV, D).npu().to(fp8)
    v = torch.randn(nb, BLOCK, NKV, D).npu().to(fp8)
    q_descale = scales((seq_len, NQ, D // WINDOW, 2))
    k_descale = scales((nb, BLOCK, NKV, D // WINDOW, 2))
    v_descale = scales((nb, BLOCK // WINDOW, NKV, D, 2))

    common = {
        "cu_seqlens_q": torch.tensor([0, seq_len], dtype=torch.int32).npu(),
        "seqused_kv": torch.tensor([seq_len], dtype=torch.int32).npu(),
        "mask_mode": 3,
        "max_seqlen_q": seq_len,
        "max_seqlen_kv": seq_len,
        "layout_q": "TND",
        "layout_q_descale": "TND",
        "layout_kv": "PA_BBND",
        "layout_out": "TND",
    }
    label = f"len={seq_len} table_cols={cols} (blocks={blocks})"
    print(f"  [RUN] {label}", flush=True)

    metadata = torch.ops._C_ascend.npu_quant_flash_attn_metadata(
        NQ, NKV, D, 1, v_descale=v_descale, **common)
    torch.npu.synchronize()
    print(f"  metadata buffer: {metadata.numel()} int32", flush=True)

    out, _ = torch.ops._C_ascend.npu_quant_flash_attn(
        q, k, v, q_descale, k_descale, v_descale, 1,
        block_table=table.npu(), attn_mask=causal_mask_npu(), metadata=metadata,
        softmax_scale=D ** -0.5, **common)
    torch.npu.synchronize()
    print(f"  attn_out: shape={tuple(out.shape)} finite={bool(out.float().isfinite().all())}",
          flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", type=int)
    ap.add_argument("--cols", type=int, default=0)
    args = ap.parse_args()

    if args.len is not None:
        return run_one(args.len, args.cols)

    combos = [(n, 0) for n in SWEEP_LENS] + [(SWEEP_LENS[-1], VLLM_TABLE_COLS)]
    results = []
    for seq_len, cols in combos:
        label = f"len={seq_len:>5} cols={'tight' if not cols else cols}"
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--len", str(seq_len), "--cols", str(cols)],
            capture_output=True, text=True)
        ok = proc.returncode == 0
        results.append((label, ok))
        print(f"[{'GREEN' if ok else 'RED'}] {label}")
        if not ok:
            tail = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.strip()][-6:]
            for line in tail:
                print(f"       {line}")

    print("\n== summary ==")
    for label, ok in results:
        print(f"  {label}: {'GREEN' if ok else 'RED'}")
    tight = [ok for (label, ok) in results if "tight" in label]
    padded = [ok for (label, ok) in results if "tight" not in label]
    print("\n== reading ==")
    if not all(tight):
        first_bad = next(lbl for lbl, ok in results if not ok)
        print(f"  Length-dependent abort ({first_bad.strip()}) -> hypothesis A:"
              " the metadata split-plan buffer is undersized.")
    elif padded and not all(padded):
        print("  Only the vLLM-shaped padded table aborts -> hypothesis B:"
              " the zero padding in block_table is what the op rejects.")
    else:
        print("  Everything survived: neither hypothesis reproduces here; the abort"
              " needs something else from the live engine (per-layer cache size,"
              " real block ids, concurrent streams).")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
