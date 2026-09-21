#!/usr/bin/env python3
"""Does quant_flash_attn_metadata accept a view of the V scale cache as v_descale?

junlin-c8-mxfp-16614 stopped allocating a zero stub for the metadata op's
v_descale every step and hands it the first two bytes of the layer's own
PA_NZ V scale cache instead (AscendC8MXFPAttentionBackendImpl.
_qfa_v_descale_placeholder). The shape, dtype and strides are identical to the
old stub; what changed is that the bytes are not zero and the tensor is a view
into a much larger storage. Neither can be checked without the operator, and
finding out from a 2.4T service start costs a full weight load.

This runs on one NPU in seconds, loads no weights, and answers two things:

  1. the operator's entry check accepts the view            (else: it raises)
  2. the plan does not depend on v_descale's bytes or origin (else: plans differ)

Usage (inside the container):
    python3 scripts/debug/qfa_metadata_vdescale_view.py
    QFA_DEVICE=3 python3 scripts/debug/qfa_metadata_vdescale_view.py

GREEN on every line means the change is safe to serve. Any RED: revert
_qfa_v_descale_placeholder to the torch.zeros stub and send the output back.
Delete this script once that is settled.
"""

from __future__ import annotations

import os
import sys

import torch

# 2.4T at TP8: 64 query heads and 4 KV heads over 8 ranks -> 8 / 1 per rank.
NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE = 8, 1, 256
BLOCK_SIZE, NUM_BLOCKS = 512, 16
QUANT_MODE_MXFP8 = 1

# (name, cu_seqlens_q, seqused_kv, max_seqlen_q, mask_mode, layout_q_descale)
CASES = [
    ("decode", [0, 1, 2, 3, 4], [10, 600, 33, 1], 1, 0, "N2TGD"),
    ("mtp-verify", [0, 2, 4, 6], [10, 600, 33], 2, 3, "N2TGD"),
    ("prefill", [0, 700, 1500], [700, 1300], 800, 3, "TND"),
]


def flatten(plan) -> list[torch.Tensor]:
    if isinstance(plan, torch.Tensor):
        return [plan]
    if isinstance(plan, (list, tuple)):
        return [t for item in plan for t in flatten(item)]
    return []


def main() -> int:
    try:
        import torch_npu  # noqa: F401
        from cann_ops_transformer.ops import quant_flash_attn_metadata
    except ImportError as exc:
        print(f"[RED ] cannot import the operator: {exc} -- run this on the server, inside the container")
        return 1

    device = torch.device(f"npu:{int(os.environ.get('QFA_DEVICE', '0'))}")
    torch.npu.set_device(device)

    # PA_NZ V scale cache: [blocks, kv heads, head_dim // 16, block_size // 64, 16, 2].
    # 127 is E8M0 for 1.0 -- deliberately not zero, unlike the stub it replaces.
    value_scale_cache = torch.full(
        (NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE // 16, BLOCK_SIZE // 64, 16, 2), 127, dtype=torch.uint8, device=device
    )
    placeholders = {
        "zeros stub (old)": torch.zeros(1, 1, 1, 1, 1, 2, dtype=torch.uint8, device=device),
        "cache view (new)": value_scale_cache.view(-1)[:2].view(1, 1, 1, 1, 1, 2),
    }

    ok = True
    for name, cu_seqlens_q, seqused_kv, max_seqlen_q, mask_mode, layout_q_descale in CASES:
        plans = {}
        for label, placeholder in placeholders.items():
            try:
                plan = quant_flash_attn_metadata(
                    NUM_HEADS,
                    NUM_KV_HEADS,
                    HEAD_SIZE,
                    QUANT_MODE_MXFP8,
                    cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device),
                    cu_seqlens_kv=None,
                    seqused_q=None,
                    seqused_kv=torch.tensor(seqused_kv, dtype=torch.int32, device=device),
                    v_descale=placeholder.view(torch.float8_e8m0fnu),
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=-1,
                    mask_mode=mask_mode,
                    win_left=-1,
                    win_right=-1,
                    layout_q="TND",
                    layout_q_descale=layout_q_descale,
                    layout_kv="PA_NZ",
                    layout_out="TND",
                )
                torch.npu.synchronize()
                plans[label] = [t.cpu() for t in flatten(plan)]
            except Exception as exc:  # noqa: BLE001 - the message is the finding
                ok = False
                print(f"[RED ] {name:10s} {label}: the operator raised {type(exc).__name__}: {exc}")

        if len(plans) < len(placeholders):
            continue
        old, new = plans["zeros stub (old)"], plans["cache view (new)"]
        if not old:
            print(f"[RED ] {name:10s} the plan holds no tensor to compare ({type(plan).__name__}); inspect it by hand")
            ok = False
        elif len(old) == len(new) and all(torch.equal(a, b) for a, b in zip(old, new)):
            sizes = [tuple(t.shape) for t in old]
            print(f"[GREEN] {name:10s} both placeholders accepted, plans byte-identical {sizes}")
        else:
            ok = False
            print(f"[RED ] {name:10s} the plan changed with the placeholder -- the operator reads v_descale")

    print(f"\n{'[GREEN] the cache view is a safe v_descale placeholder' if ok else '[RED ] keep the zeros stub'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
