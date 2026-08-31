#!/usr/bin/env python3
"""1:1 port of the two official quant_flash_attn doc examples onto the
vendored junlin-qfa API (torch.ops._C_ascend.npu_quant_flash_attn).

Source of the two examples: ops-transformer
torch_extension/cann_ops_transformer/docs/zh/quant_flash_attn.md "调用示例"
(1) TND non-PA mask_mode=0, (2) TND + PA_BNBD causal.

The doc examples target the standalone `cann_ops_transformer` torch_extension
package, which is a DIFFERENT delivery surface from the vllm-ascend vendored
build (different namespace, different install, JIT-compiled binding). This
script keeps the examples' shapes, layouts and attributes and only swaps the
entry point, so a GREEN run proves the vendored integration covers the
documented scenarios. Every case runs npu_quant_flash_attn_metadata first and
the main op second with an identical argument set (the op does no cross-call
validation), mirrors the doc's shape/dtype/isfinite asserts, and adds a
CPU-golden numeric compare.

Deliberate deviations from the doc text, all load-bearing:

  1. No `is_grad_enabled=` kwarg -- that is a torch_npu aclnn-wrapper
     convenience; the _C_ascend schema has no such keyword (it produced the
     original RuntimeError: Unknown keyword argument 'is_grad_enabled').
  2. descale tensors are real data-derived e8m0 scales built as
     uint8-packed -> view(torch.float8_e8m0), instead of
     torch.randn(..., dtype=torch.float8_e8m0). quant_mode=1 requires E8M0
     (see csrc/attention/quant_flash_attn/op_api/aclnn_quant_flash_attn.h),
     and random e8m0 bytes hit the 0xFF NaN sentinel / huge exponents, which
     flaky-fails the doc's own isfinite() assert. float32 descale is wrong
     for quant_mode=1 (it belongs to the per-tensor mode).
  3. Example 2's causal attn_mask uses triu(diagonal=1) int8 (1 = masked
     future) like the official test assets; the doc's tril contradicts the
     shipped tests.
  4. A CPU golden compare is added on top of the doc asserts (the doc only
     checks shape/dtype/isfinite and feeds the op random garbage scales).

Usage (inside the serving container):
  python scripts/debug/run_doc_examples_qfa_npu.py
  DOC_CASE=1 python scripts/debug/run_doc_examples_qfa_npu.py   # TND only
  DOC_CASE=2 python scripts/debug/run_doc_examples_qfa_npu.py   # PA only
"""

import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_junlin_qfa_npu import (  # noqa: E402
    QuantSeq,
    _pack_pa_data,
    bootstrap_ops,
    causal_mask_npu,
    compare,
    e8m0_npu,
)

FP8_DTYPE = torch.float8_e4m3fn
OUT_DTYPE = torch.bfloat16


def run_case1_tnd() -> bool:
    """Doc example 1: TND non-PA, mask_mode=0, B=2 Q_S=16 KV_S=16 Nq=8 Nkv=2 D=128."""
    print("== case 1 (doc example 1): TND non-PA B=2 S=16 Nq=8 Nkv=2 D=128 "
          "mask_mode=0 ==")
    torch.manual_seed(2024)
    B, Q_S, KV_S, Q_N, KV_N, D = 2, 16, 16, 8, 2, 128
    softmax_scale = 1.0 / (D ** 0.5)

    quants, goldens = [], []
    for _ in range(B):
        q = torch.randn(Q_S, Q_N, D, dtype=torch.bfloat16)
        k = torch.randn(KV_S, KV_N, D, dtype=torch.bfloat16)
        v = torch.randn(KV_S, KV_N, D, dtype=torch.bfloat16)
        qs = QuantSeq(q, k, v)
        quants.append(qs)
        goldens.append(qs.golden(softmax_scale, mask_mode=0))

    cu_q = torch.tensor([0, Q_S, Q_S * 2], dtype=torch.int32).npu()
    cu_kv = torch.tensor([0, KV_S, KV_S * 2], dtype=torch.int32).npu()
    v_descale = e8m0_npu(torch.cat([qs.v_scale_packed_rows() for qs in quants]))

    metadata = torch.ops._C_ascend.npu_quant_flash_attn_metadata(
        Q_N, KV_N, D, 1,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        v_descale=v_descale,
        batch_size=B,
        max_seqlen_q=Q_S,
        max_seqlen_kv=KV_S,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        layout_q="TND",
        layout_q_descale="TND",
        layout_kv="TND",
        layout_out="TND",
    )

    attn_out, softmax_lse = torch.ops._C_ascend.npu_quant_flash_attn(
        torch.cat([qs.q_fp8 for qs in quants]).npu(),
        torch.cat([qs.k_fp8 for qs in quants]).npu(),
        torch.cat([qs.v_fp8 for qs in quants]).npu(),
        e8m0_npu(torch.cat([qs.qk_scale_packed_tnd("q") for qs in quants])),
        e8m0_npu(torch.cat([qs.qk_scale_packed_tnd("k") for qs in quants])),
        v_descale,
        1,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        metadata=metadata,
        softmax_scale=softmax_scale,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        layout_q="TND",
        layout_q_descale="TND",
        layout_kv="TND",
        layout_out="TND",
        return_softmax_lse=False,
    )
    torch.npu.synchronize()
    # Doc asserts, verbatim:
    assert attn_out.shape == (B * Q_S, Q_N, D)
    assert attn_out.dtype == OUT_DTYPE
    assert torch.isfinite(attn_out.float()).all().item()
    assert softmax_lse.numel() == 0  # return_softmax_lse=False -> empty Tensor
    print("  doc asserts (shape/dtype/isfinite/empty-lse): [GREEN]")
    return compare("case1", attn_out, torch.cat(goldens))


def run_case2_pa() -> bool:
    """Doc example 2: TND + PA_BNBD causal, B=2 Q_S=16 KV_S=16 Nq=8 Nkv=2 D=128 block=512."""
    print("== case 2 (doc example 2): PA_BNBD causal B=2 S=16 Nq=8 Nkv=2 D=128 "
          "block=512 mask_mode=3 ==")
    torch.manual_seed(2025)
    B, Q_S, KV_S, Q_N, KV_N, D = 2, 16, 16, 8, 2, 128
    pa_block_size = 512

    data = _pack_pa_data([Q_S] * B, [KV_S] * B, Q_N, KV_N, D, pa_block_size,
                         "PA_BNBD")

    v_descale = e8m0_npu(data["v_scale_cache"])
    cu_q = data["cu_seqlens_q"].npu()
    seqused_kv = data["seqused_kv"].npu()

    metadata = torch.ops._C_ascend.npu_quant_flash_attn_metadata(
        Q_N, KV_N, D, 1,
        cu_seqlens_q=cu_q,
        seqused_kv=seqused_kv,
        v_descale=v_descale,
        batch_size=B,
        max_seqlen_q=Q_S,
        max_seqlen_kv=KV_S,
        mask_mode=3,
        win_left=-1,
        win_right=-1,
        layout_q="TND",
        layout_q_descale="TND",
        layout_kv="PA_BNBD",
        layout_out="TND",
    )

    attn_out, softmax_lse = torch.ops._C_ascend.npu_quant_flash_attn(
        data["q"].npu(),
        data["k_cache"].npu().view(FP8_DTYPE),
        data["v_cache"].npu().view(FP8_DTYPE),
        e8m0_npu(data["q_scale_packed_tnd"]),
        e8m0_npu(data["k_scale_cache"]),
        v_descale,
        1,
        block_table=data["block_table"].npu(),
        cu_seqlens_q=cu_q,
        seqused_kv=seqused_kv,
        attn_mask=causal_mask_npu(),
        metadata=metadata,
        softmax_scale=data["softmax_scale"],
        mask_mode=3,
        win_left=-1,
        win_right=-1,
        layout_q="TND",
        layout_q_descale="TND",
        layout_kv="PA_BNBD",
        layout_out="TND",
        return_softmax_lse=False,
    )
    torch.npu.synchronize()
    # Doc asserts, verbatim:
    assert attn_out.shape == (B * Q_S, Q_N, D)
    assert attn_out.dtype == OUT_DTYPE
    assert torch.isfinite(attn_out.float()).all().item()
    assert softmax_lse.numel() == 0
    print("  doc asserts (shape/dtype/isfinite/empty-lse): [GREEN]")
    return compare("case2", attn_out, torch.cat(data["goldens"]))


CASES = {"1": run_case1_tnd, "2": run_case2_pa}


def main() -> int:
    import torch_npu  # noqa: F401

    torch.npu.set_device(int(os.environ.get("QFA_DEVICE", "0")))
    bootstrap_ops()

    selected = [n.strip() for n in os.environ.get("DOC_CASE", "1,2").split(",")]
    results = {}
    for name in selected:
        if name not in CASES:
            print(f"[RED] unknown case {name}; known: {sorted(CASES)}")
            return 2
        try:
            results[name] = CASES[name]()
        except Exception:
            traceback.print_exc()
            results[name] = False
        torch.npu.synchronize()

    print("\n== summary ==")
    all_green = True
    for name, ok in results.items():
        print(f"  doc example {name}: {'GREEN' if ok else 'RED'}")
        all_green &= ok
    print(f"[{'GREEN' if all_green else 'RED'}] doc quant_flash_attn examples "
          f"on torch.ops._C_ascend (vendored junlin-qfa)")
    return 0 if all_green else 1


if __name__ == "__main__":
    sys.exit(main())
