#!/usr/bin/env python3
"""CPU-only parameter test for the QFA backend's forward dispatch (M2 split).

Stubs torch.ops._C_ascend with recorders and drives forward_impl with fake
metadata, asserting the decode/prefill split hands the operator exactly the
slices the FIA split precedent defines:

  MIXED      decode half first (N2TGD descale, own cu/seqused/block rows),
             prefill half rebased to 0 (TND), outputs concatenated in order
  DECODE     one call, N2TGD when G*max_q <= 80
  PREFILL    one call, TND
  WIDE-Q     G*max_q > 80 falls back to TND even for pure decode

Run:  QFA_WORKTREE=junlin-qfa-m2 python scripts/tests/test_qfa_backend_forward_split.py
"""

import os
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QFA_WORKTREE", "junlin-qfa-m2")
from test_qfa_backend_write_path import (  # noqa: E402
    QFA_PY, alloc_cache, check, load_qfa_module, make_impl)


class Recorder:
    def __init__(self):
        self.calls = []

    def npu_quant_flash_attn_metadata(self, *args, **kwargs):
        return torch.zeros(4096, dtype=torch.int32)

    def npu_quant_flash_attn(self, q, k, v, q_descale, k_descale, v_descale,
                             quant_mode, **kwargs):
        self.calls.append({
            "q_shape": tuple(q.shape),
            "descale_shape": tuple(q_descale.shape),
            "layout_q_descale": kwargs["layout_q_descale"],
            "cu_seqlens_q": kwargs["cu_seqlens_q"].tolist(),
            "seqused_kv": kwargs["seqused_kv"].tolist(),
            "block_rows": kwargs["block_table"].shape[0],
            "max_seqlen_q": kwargs["max_seqlen_q"],
            "max_seqlen_kv": kwargs["max_seqlen_kv"],
        })
        t = q.shape[0]
        return (torch.zeros(t, q.shape[1], q.shape[2], dtype=torch.bfloat16),
                torch.zeros(0))


def make_meta(qfa, q_lens_decode, q_lens_prefill, kv_lens):
    q_lens = q_lens_decode + q_lens_prefill
    cu = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32)
    n = len(kv_lens)
    meta = types.SimpleNamespace(
        attn_state=qfa.AscendAttentionState.ChunkedPrefill,
        causal=True,
        num_actual_tokens=int(cu[-1]),
        query_start_loc=cu,
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32),
        seq_lens_cpu=torch.tensor(kv_lens, dtype=torch.int32),
        seq_lens_list=list(kv_lens),
        actual_seq_lengths_q=cu[1:].tolist(),  # builder drops the leading 0
        block_tables=torch.arange(n * 4, dtype=torch.int32).reshape(n, 4),
        max_query_len=max(q_lens),
        num_decodes=len(q_lens_decode),
        num_prefills=len(q_lens_prefill),
        num_decode_tokens=sum(q_lens_decode),
    )
    return meta


def run_case(qfa, rec, nq, nkv, d, q_lens_decode, q_lens_prefill, kv_lens):
    rec.calls.clear()
    impl = make_impl(qfa, nkv, d)
    impl.num_heads = nq
    impl.scale = 1.0
    impl._ensure_planes(alloc_cache(qfa, len(kv_lens) * 4, 128, nkv, d))
    meta = make_meta(qfa, q_lens_decode, q_lens_prefill, kv_lens)
    total = meta.num_actual_tokens
    query = torch.randn(total, nq, d, dtype=torch.bfloat16)
    output = torch.zeros(total, nq * d, dtype=torch.bfloat16)
    impl.forward_impl(query, None, None, None, meta, output)
    return rec.calls


def main() -> int:
    torch.manual_seed(1)
    qfa = load_qfa_module()
    rec = Recorder()
    torch.ops._C_ascend = rec  # process-local stub

    nq, nkv, d = 24, 4, 256  # G = 6
    all_ok = True

    # ---- MIXED: 2 decodes (q=1) + 2 prefills (q=50, 30) ----
    calls = run_case(qfa, rec, nq, nkv, d, [1, 1], [50, 30], [300, 40, 50, 30])
    ok = len(calls) == 2
    if ok:
        dec, pre = calls
        ok &= dec["q_shape"][0] == 2 and dec["layout_q_descale"] == "N2TGD"
        ok &= dec["descale_shape"] == (nkv, 2, 6, d // 64, 2)
        ok &= dec["cu_seqlens_q"] == [0, 1, 2] and dec["seqused_kv"] == [300, 40]
        ok &= dec["block_rows"] == 2 and dec["max_seqlen_q"] == 1
        ok &= dec["max_seqlen_kv"] == 300
        ok &= pre["q_shape"][0] == 80 and pre["layout_q_descale"] == "TND"
        ok &= pre["cu_seqlens_q"] == [0, 50, 80] and pre["seqused_kv"] == [50, 30]
        ok &= pre["block_rows"] == 2 and pre["max_seqlen_q"] == 50
        ok &= pre["max_seqlen_kv"] == 50
    else:
        print(f"    expected 2 calls, got {len(calls)}")
    all_ok &= check("MIXED-SPLIT", ok)

    # ---- DECODE only, spec verify q=4 uniform (G*4 = 24 <= 80) ----
    calls = run_case(qfa, rec, nq, nkv, d, [4, 4, 4], [], [70, 130, 260])
    ok = (len(calls) == 1 and calls[0]["layout_q_descale"] == "N2TGD"
          and calls[0]["q_shape"][0] == 12
          and calls[0]["descale_shape"] == (nkv, 12, 6, d // 64, 2)
          and calls[0]["cu_seqlens_q"] == [0, 4, 8, 12]
          and calls[0]["max_seqlen_kv"] == 260)
    all_ok &= check("DECODE-N2TGD", ok)

    # ---- PREFILL only ----
    calls = run_case(qfa, rec, nq, nkv, d, [], [60, 40], [60, 40])
    ok = (len(calls) == 1 and calls[0]["layout_q_descale"] == "TND"
          and calls[0]["q_shape"][0] == 100
          and calls[0]["cu_seqlens_q"] == [0, 60, 100])
    all_ok &= check("PREFILL-TND", ok)

    # ---- WIDE-Q: G * max_q > 80 falls back to TND ----
    calls = run_case(qfa, rec, 128, 4, d, [16, 16], [], [70, 90])  # G=32, 32*16>80
    ok = len(calls) == 1 and calls[0]["layout_q_descale"] == "TND"
    all_ok &= check("WIDE-Q-TND-FALLBACK", ok)

    print(f"[{'GREEN' if all_ok else 'RED'}] QFA forward split dispatch "
          f"(worktree {QFA_PY.parents[2].name})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
