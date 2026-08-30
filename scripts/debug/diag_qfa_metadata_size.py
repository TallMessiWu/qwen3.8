#!/usr/bin/env python3
"""Pin down why the QFA metadata op aborts on a real 27B prefill.

A 1594-token prefill kills the AICPU metadata kernel (errcode 0x2a,
kernelName=QuantFlashAttnMetadata) while the single-op smoke script stays green.
An engine-side dump showed every argument the op receives is identical to what
this script already reproduced green -- num_reqs=1, cu_seqlens_q=[0, 1594],
seqused_kv=[1594], max_seqlen 1594, Nq=24 Nkv=4 D=256, block_table (1, 1044) with
24 live entries -- with one exception: the KV cache holds 9672 blocks, not 14.

That matters because the five planes are carved out of one engine-allocated
buffer, so a bigger cache pushes each plane's byte offset further out:

    offset(v_scale) = num_blocks * (block*N*D*2 + block*N*(D/64)*2)
                    = num_blocks * 266240

which crosses the int32 ceiling (2^31) at num_blocks = 8066. The live cache sits
at 9672 (2.58 GB in), this script used to sit at 14, and v_descale is the one
large tensor the metadata op is handed. k_fp8 (offset 0), k_scale (1.27 GB) and
v_fp8 (1.31 GB) all stay under the line -- only v_scale crosses it.

So sweep num_blocks across that boundary, with the planes carved exactly the way
attention_qfa._ensure_planes does it, and print each plane's real byte offset.
A clean break either side of ~8066 indicts the offset; failures unrelated to it
point elsewhere.

Each combination runs in its own subprocess -- an AICPU abort poisons the device
for everything that follows, so a shared process could only ever report the first
failure.

No weights, no golden comparison: this only asks whether the launch survives.

Usage (inside the serving container):
  python scripts/debug/diag_qfa_metadata_size.py
  python scripts/debug/diag_qfa_metadata_size.py --len 1594 --blocks 9672  # one combo
"""

import argparse
import os
import subprocess
import sys

NQ, NKV, D = 24, 4, 256  # Qwen3.8-27B per-rank head shape
BLOCK = 128
WINDOW = 64
SEQ_LEN = 1594  # what the live prefill asked for
LIVE_BLOCKS = 9672  # what the live KV cache holds
LIVE_TABLE_COLS = 1044
INT32_MAX = 2 ** 31
# offset(v_scale) = num_blocks * V_SCALE_STRIDE, so this is where it crosses 2^31
V_SCALE_STRIDE = BLOCK * NKV * D * 2 + BLOCK * NKV * (D // WINDOW) * 2
BOUNDARY = INT32_MAX // V_SCALE_STRIDE  # 8065
SWEEP_BLOCKS = [64, BOUNDARY - 64, BOUNDARY + 64, LIVE_BLOCKS]


def carve_planes(num_blocks: int):
    """Carve the five planes out of one buffer, exactly like _ensure_planes."""
    import torch

    nb, bs, n, d = num_blocks, BLOCK, NKV, D
    sizes = [
        nb * bs * n * d,                    # K fp8
        nb * bs * n * (d // WINDOW) * 2,    # K scale
        nb * bs * n * d,                    # V fp8
        nb * (bs // WINDOW) * n * d * 2,    # V scale
        nb * bs * n * d * 2,                # V staging (bf16)
    ]
    flat = torch.zeros(sum(sizes), dtype=torch.uint8).npu()
    offsets, acc = [], 0
    for size in sizes:
        offsets.append(acc)
        acc += size
    seg = [flat[offsets[i]: offsets[i] + sizes[i]] for i in range(5)]
    planes = {
        "k_fp8": seg[0].reshape(nb, bs, n, d),
        "k_scale": seg[1].reshape(nb, bs, n, d // WINDOW, 2),
        "v_fp8": seg[2].reshape(nb, bs, n, d),
        "v_scale": seg[3].reshape(nb, bs // WINDOW, n, d, 2),
    }
    return planes, dict(zip(["k_fp8", "k_scale", "v_fp8", "v_scale", "v_stage"], offsets))


def run_one(seq_len: int, table_cols: int, num_blocks: int) -> int:
    import torch
    import torch_npu  # noqa: F401

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_junlin_qfa_npu import bootstrap_ops, causal_mask_npu

    torch.npu.set_device(int(os.environ.get("QFA_DEVICE", "0")))
    bootstrap_ops()

    fp8, e8m0 = torch.float8_e4m3fn, torch.float8_e8m0fnu
    live_blocks = (seq_len + BLOCK - 1) // BLOCK
    num_blocks = max(num_blocks, live_blocks + 1)
    cols = table_cols or live_blocks
    # Block ids start at 1 so every used entry is a positive integer.
    table = torch.zeros(1, cols, dtype=torch.int32)
    table[0, :live_blocks] = torch.arange(1, live_blocks + 1, dtype=torch.int32)

    planes, offsets = carve_planes(num_blocks)
    print(f"  [RUN] len={seq_len} cols={cols} blocks={num_blocks}", flush=True)
    for name, off in offsets.items():
        flag = "  <-- past int32" if off >= INT32_MAX else ""
        print(f"    offset({name}) = {off} bytes{flag}", flush=True)

    q = torch.randn(seq_len, NQ, D).npu().to(fp8)
    q_descale = torch.full((seq_len, NQ, D // WINDOW, 2), 127, dtype=torch.uint8).npu().view(e8m0)
    v_descale = planes["v_scale"].view(e8m0)

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

    metadata = torch.ops._C_ascend.npu_quant_flash_attn_metadata(
        NQ, NKV, D, 1, v_descale=v_descale, **common)
    torch.npu.synchronize()
    print(f"  metadata OK: {metadata.numel()} int32", flush=True)

    out, _ = torch.ops._C_ascend.npu_quant_flash_attn(
        q, planes["k_fp8"].view(fp8), planes["v_fp8"].view(fp8),
        q_descale, planes["k_scale"].view(e8m0), v_descale, 1,
        block_table=table.npu(), attn_mask=causal_mask_npu(), metadata=metadata,
        softmax_scale=D ** -0.5, **common)
    torch.npu.synchronize()
    print(f"  main op OK: shape={tuple(out.shape)}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", type=int)
    ap.add_argument("--cols", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=0)
    args = ap.parse_args()

    if args.len is not None:
        return run_one(args.len, args.cols, args.blocks)

    combos = [(SEQ_LEN, LIVE_TABLE_COLS, nb) for nb in SWEEP_BLOCKS]
    print(f"v_scale crosses int32 at num_blocks = {BOUNDARY + 1}"
          f" (stride {V_SCALE_STRIDE} bytes/block)")
    print()
    results = []
    for seq_len, cols, blocks in combos:
        off = blocks * V_SCALE_STRIDE
        label = (f"blocks={blocks:>5} v_scale_offset={off / 2**30:5.2f}GB "
                 f"{'>=2^31' if off >= INT32_MAX else ' <2^31'}")
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--len", str(seq_len),
             "--cols", str(cols), "--blocks", str(blocks)],
            capture_output=True, text=True)
        ok = proc.returncode == 0
        results.append((label, ok, off))
        print(f"[{'GREEN' if ok else 'RED'}] {label}")
        if not ok:
            tail = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.strip()][-5:]
            for line in tail:
                print(f"       {line}")

    print()
    print("== summary ==")
    for label, ok, _ in results:
        print(f"  {label}: {'GREEN' if ok else 'RED'}")

    print()
    print("== reading ==")
    under = [ok for _, ok, off in results if off < INT32_MAX]
    over = [ok for _, ok, off in results if off >= INT32_MAX]
    if under and over and all(under) and not any(over):
        print("  Clean break at the int32 line -> the plane's byte offset is the"
              " problem, not the sequence, the block table or the plan size.")
    elif not any(under + over):
        print("  Everything aborts, including small caches -> the offset is not"
              " what matters; something else in this setup differs from the"
              " single-op script.")
    elif all(under + over):
        print("  Everything survived -> a big cache alone does not reproduce it;"
              " the live engine must differ in something this script still fakes"
              " (real quantized data, per-layer reuse, concurrent streams).")
    else:
        print("  Mixed result that does not line up with the int32 boundary --"
              " read the per-case offsets above before concluding.")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
