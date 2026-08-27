#!/usr/bin/env python3
"""Can a V window be re-quantized from its own committed FP8 instead of from
the original BF16 the staging ring keeps?

If yes, the ring goes away - and with it the bug that it is indexed by
position in the batch, which vLLM reassigns whenever a request finishes or the
builder reorders. Fixing that index without dropping the ring would mean
gathering the whole ring every step (no cheaper test is possible without a
device-to-host sync), which at max_num_seqs=64 is ~267 MB of copies per step
against the ~131 KB a decode step actually writes.

The claim: an E8M0 scale is a power of two, so raising it only shifts the
exponent, and a value already in E4M3 is still exactly representable at the
new scale. Re-quantizing from FP8 should therefore produce byte-identical
output to re-quantizing from the originals. Verified in fp32 on a GPU already;
this checks it against the real npu_dynamic_mx_quant and, just as importantly,
that the dequantize step has kernels at all - float8 has no transpose or
index_put_ on this hardware, and the AICPU fallback aborts the stream.

Run (seconds, no server):
    python scripts/debug/test_qfa_requant_npu.py
"""

import os
import sys

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("[RED] torch_npu unavailable - run this on the server")
    sys.exit(2)

GROUP = 64  # tokens per V window
PACK = 32  # tokens sharing one E8M0 scale; two packed per window
NUM_KV_HEADS = int(os.environ.get("NKV", "4"))
HEAD_SIZE = int(os.environ.get("D", "256"))
DEVICE = "npu"


def quant_along_tokens(rows: torch.Tensor):
    """Mirror of _qfa_quant_along_tokens: (W, 64, N, D) -> fp8 bytes + scales."""
    w, group, n, d = rows.shape
    cols = rows.permute(0, 2, 3, 1).reshape(w * n * d, group)
    fp8, scale = torch_npu.npu_dynamic_mx_quant(cols.contiguous(), dst_type=torch.float8_e4m3fn)
    fp8 = fp8.view(torch.uint8).reshape(w, n, d, group).permute(0, 3, 1, 2)
    scale = scale.view(torch.uint8).reshape(w, n, d, 2)
    return fp8.contiguous(), scale


def dequant_windows(fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """(W,64,N,D) uint8 + (W,N,D,2) e8m0 -> BF16, the read-back under test.

    E8M0 byte b encodes 2^(b-127). Each packed pair covers PACK tokens, so the
    factor is expanded along the token axis before it multiplies the values.
    """
    w, group, n, d = fp8.shape
    vals = fp8.view(torch.float8_e4m3fn).to(torch.bfloat16)
    exp = scale.to(torch.int32) - 127
    factor = torch.exp2(exp.to(torch.float32)).to(torch.bfloat16)  # (W,N,D,2)
    factor = factor.permute(0, 3, 1, 2).repeat_interleave(PACK, dim=1)  # (W,64,N,D)
    return vals * factor


def main() -> int:
    torch.manual_seed(1024)
    ok = True

    for label, scale_profile in (
        ("flat magnitudes", torch.ones(GROUP)),
        ("magnitudes growing 64x", torch.exp2(torch.linspace(0, 6, GROUP))),
        ("late 50x outlier", torch.cat([torch.ones(GROUP - 1), torch.tensor([50.0])])),
    ):
        vals = torch.randn(1, GROUP, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.bfloat16)
        vals = (vals * scale_profile.view(1, GROUP, 1, 1)).to(torch.bfloat16).to(DEVICE)

        # Step 1: the first half is already committed to the cache.
        half = vals.clone()
        half[:, GROUP // 2 :] = 0
        fp8_half, scale_half = quant_along_tokens(half)

        # Step 2a: today - re-quantize the full window from the originals.
        fp8_ref, scale_ref = quant_along_tokens(vals)

        # Step 2b: proposed - history read back from FP8, new tokens original.
        recovered = dequant_windows(fp8_half, scale_half)
        merged = recovered.clone()
        merged[:, GROUP // 2 :] = vals[:, GROUP // 2 :]
        fp8_new, scale_new = quant_along_tokens(merged)

        same_scale = bool(torch.equal(scale_ref, scale_new))
        same_bytes = bool(torch.equal(fp8_ref, fp8_new))
        drift = (fp8_ref.to(torch.int16) - fp8_new.to(torch.int16)).abs()
        print(f"== {label}")
        print(f"   scales identical: {same_scale}   value bytes identical: {same_bytes}")
        if not same_bytes:
            print(f"   differing bytes: {int((drift > 0).sum())}/{drift.numel()}, max step {int(drift.max())}")
        ok = ok and same_scale and same_bytes

    # The history the op never wrote must read back as +0.0, or the padded tail
    # past seqused_kv stops being benign.
    zeros = torch.zeros(1, GROUP, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device=DEVICE)
    fp8_z, scale_z = quant_along_tokens(zeros)
    back = dequant_windows(fp8_z, scale_z)
    zero_ok = bool(torch.equal(back, zeros))
    print(f"== all-zero window round-trips to zero: {zero_ok}")
    ok = ok and zero_ok

    print(f"[{'GREEN' if ok else 'RED'}] re-quantizing from committed FP8 is lossless")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
