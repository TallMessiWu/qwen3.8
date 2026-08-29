#!/usr/bin/env python3
"""CPU end-to-end simulation of the M1 QFA backend under a real engine schedule.

Drives the backend's own reshape_and_cache + forward_impl through the step
pattern a live engine produces - solo prefill, decode, mixed prefill+decode,
MTP verify (q=4), rejection rollback - with the NPU pieces replaced by
semantically exact CPU stand-ins:

  npu_dynamic_mx_quant   -> CPU implementation (byte-identical, QUANT-FEED)
  npu_quant_flash_attn   -> a stand-in that consumes the REAL cache planes
                            with the operator's PA_BBND semantics (gather via
                            block_table, unpack e8m0 scales, run the official
                            golden pipeline) - so cache layout, write path and
                            read semantics are exercised end to end
  npu_quant_flash_attn_metadata -> zeros (split plan is a device-side detail)

Per step:
  L1  cache planes byte-equal a from-scratch reference packing of every
      request's full history (write-path integrity under the schedule)
  L2  attention output close to a bf16 fp32-reference attention (loose
      threshold: quantization noise passes, misindexing fails hard)

Run:  python scripts/tests/test_qfa_backend_e2e_sim.py            (M1)
      QFA_WORKTREE=junlin-qfa-m2 python scripts/tests/...         (M2/M3)
"""

import math
import os
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QFA_WORKTREE", "junlin-qfa")
from test_qfa_backend_write_path import (  # noqa: E402
    alloc_cache, check, load_golden_module, load_qfa_module, make_impl)

WINDOW = 64


# ---------------------------------------------------------------------------
# QFA main-op stand-in: real PA_BBND cache consumption + official golden math
# ---------------------------------------------------------------------------
class FakeQfaOps:
    def __init__(self, golden, nq, nkv, d, bs):
        self.g = golden
        self.nq, self.nkv, self.d, self.bs = nq, nkv, d, bs

    def npu_quant_flash_attn_metadata(self, *args, **kwargs):
        return torch.zeros(4096, dtype=torch.int32)

    def _unpack_kv_seq(self, k, v, k_descale, v_descale, block_table_row, s):
        bs, n, d = self.bs, self.nkv, self.d
        nblk = (s + bs - 1) // bs
        blocks = block_table_row[:nblk].long()
        kb = k.view(torch.uint8)[blocks].reshape(nblk * bs, n, d)[:s]
        vb = v.view(torch.uint8)[blocks].reshape(nblk * bs, n, d)[:s]
        ks = (k_descale.view(torch.uint8)[blocks]
              .reshape(nblk * bs, n, d // WINDOW, 2)[:s]
              .reshape(s, n, d // 32))
        vs_pairs = v_descale.view(torch.uint8)[blocks]  # (nblk, bs//64, n, d, 2)
        rows = vs_pairs.reshape(nblk * (bs // WINDOW), n, d, 2)
        sg_pad = rows.shape[0] * 2
        vs = torch.zeros(sg_pad, n, d, dtype=torch.uint8)
        vs[0::2] = rows[..., 0]
        vs[1::2] = rows[..., 1]
        sg = (s + 31) // 32
        e2f = self._e8m0_to_fp32
        return (kb.view(torch.float8_e4m3fn), vb.view(torch.float8_e4m3fn),
                e2f(ks).permute(1, 0, 2).unsqueeze(0),          # (1, n, s, dg)
                e2f(vs[:sg]).permute(1, 0, 2).unsqueeze(0))     # (1, n, sg, d)

    @staticmethod
    def _e8m0_to_fp32(b):
        out = torch.pow(2.0, b.to(torch.float32) - 127.0)
        return torch.where(b == 0xFF, torch.zeros_like(out), out)

    def npu_quant_flash_attn(self, q, k, v, q_descale, k_descale, v_descale,
                             quant_mode, **kw):
        assert kw["layout_kv"] == "PA_BBND", kw["layout_kv"]
        cu = kw["cu_seqlens_q"].tolist()
        seqused = kw["seqused_kv"].tolist()
        bt = kw["block_table"]
        scale = kw["softmax_scale"]
        if kw["layout_q_descale"] == "N2TGD":
            # (Nkv, T, G, dg2, 2) -> (T, Nq, dg2, 2)
            b = q_descale.view(torch.uint8)
            nkv_, t_, g_, dg2, _ = b.shape
            qd = b.permute(1, 0, 2, 3, 4).reshape(t_, nkv_ * g_, dg2, 2)
        else:
            qd = q_descale.view(torch.uint8).reshape(q.shape[0], self.nq, -1, 2)
        qd = qd.reshape(q.shape[0], self.nq, -1)  # (T, nq, dg)
        outs = []
        for i, s in enumerate(seqused):
            lo, hi = cu[i], cu[i + 1]
            kf, vf, ksc, vsc = self._unpack_kv_seq(k, v, k_descale, v_descale, bt[i], s)
            qsc = self._e8m0_to_fp32(qd[lo:hi]).permute(1, 0, 2).unsqueeze(0)
            outs.append(self.g.cpu_golden_one_seq(
                q[lo:hi], kf, vf, qsc, ksc, vsc, scale, mask_mode=3))
        out = torch.cat(outs).to(torch.bfloat16)
        return out, torch.zeros(0)


# ---------------------------------------------------------------------------
# bf16 fp32-reference attention (bottom-right causal, GQA broadcast)
# ---------------------------------------------------------------------------
def ref_attention(q_bf, k_hist, v_hist, scale):
    ql, nq, d = q_bf.shape
    s, nkv, _ = k_hist.shape
    rep = nq // nkv
    q = q_bf.float().permute(1, 0, 2)
    kk = k_hist.float().permute(1, 0, 2).repeat_interleave(rep, dim=0)
    vv = v_hist.float().permute(1, 0, 2).repeat_interleave(rep, dim=0)
    logits = torch.matmul(q, kk.transpose(-1, -2)) * scale
    qr = torch.arange(ql).view(-1, 1)
    kr = torch.arange(s).view(1, -1)
    mask = kr > (qr + (s - ql))
    logits = logits.masked_fill(mask.unsqueeze(0), float("-inf"))
    return torch.matmul(torch.softmax(logits, dim=-1), vv).permute(1, 0, 2)


# ---------------------------------------------------------------------------
# reference cache packing from the ledger (golden-module quantization)
# ---------------------------------------------------------------------------
def pack_reference(golden, ledger, block_tables, nb, bs, nkv, d):
    planes = {
        "k": torch.zeros(nb, bs, nkv, d, dtype=torch.uint8),
        "ks": torch.zeros(nb, bs, nkv, d // WINDOW, 2, dtype=torch.uint8),
        "v": torch.zeros(nb, bs, nkv, d, dtype=torch.uint8),
        "vs": torch.zeros(nb, bs // WINDOW, nkv, d, 2, dtype=torch.uint8),
    }
    for req, (k_hist, v_hist) in ledger.items():
        s = k_hist.shape[0]
        if s == 0:
            continue
        nblk = (s + bs - 1) // bs
        s_pad = nblk * bs
        k_pad = torch.nn.functional.pad(k_hist.float(), (0, 0, 0, 0, 0, s_pad - s)).bfloat16()
        v_pad = torch.nn.functional.pad(v_hist.float(), (0, 0, 0, 0, 0, s_pad - s)).bfloat16()
        qs = golden.QuantSeq(torch.zeros(1, 8, d, dtype=torch.bfloat16), k_pad, v_pad)
        kb = qs.k_fp8.view(torch.uint8)
        vb = qs.v_fp8.view(torch.uint8)
        ksb = qs.qk_scale_packed_tnd("k")
        vsb = qs.v_scale_packed_rows()
        for j in range(nblk):
            blk = int(block_tables[req][j])
            t0, t1 = j * bs, (j + 1) * bs
            planes["k"][blk] = kb[t0:t1]
            planes["v"][blk] = vb[t0:t1]
            planes["ks"][blk] = ksb[t0:t1]
            planes["vs"][blk] = vsb[j * (bs // WINDOW):(j + 1) * (bs // WINDOW)]
    return planes


# ---------------------------------------------------------------------------
# engine simulator
# ---------------------------------------------------------------------------
class EngineSim:
    def __init__(self, qfa, golden, nq, nkv, d, bs, nb):
        self.qfa, self.golden = qfa, golden
        self.nq, self.nkv, self.d, self.bs, self.nb = nq, nkv, d, bs, nb
        self.impl = make_impl(qfa, nkv, d)
        self.impl.num_heads = nq
        self.impl.scale = 1.0 / math.sqrt(d)
        self.impl.kv_sharing_target_layer_name = None
        self.cache = alloc_cache(qfa, nb, bs, nkv, d)
        self.ledger: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.block_tables: dict[str, list[int]] = {}
        self.next_block = 0

    def ensure_blocks(self, req, s):
        bt = self.block_tables.setdefault(req, [])
        while len(bt) * self.bs < s:
            bt.append(self.next_block)
            self.next_block += 1

    def rollback(self, req, accepted_len):
        k, v = self.ledger[req]
        self.ledger[req] = (k[:accepted_len], v[:accepted_len])

    def step(self, name, reqs, state):
        """reqs: ordered {req_id: q_len}; new K/V/Q are freshly sampled."""
        nkv, d, bs = self.nkv, self.d, self.bs
        q_parts, k_parts, v_parts, slots, kv_lens = [], [], [], [], []
        for req, ql in reqs.items():
            k_old, v_old = self.ledger.get(req, (torch.zeros(0, nkv, d, dtype=torch.bfloat16),) * 2)
            base = k_old.shape[0]
            k_new = torch.randn(ql, nkv, d, dtype=torch.bfloat16)
            v_new = torch.randn(ql, nkv, d, dtype=torch.bfloat16)
            q_new = torch.randn(ql, self.nq, d, dtype=torch.bfloat16)
            self.ledger[req] = (torch.cat([k_old, k_new]), torch.cat([v_old, v_new]))
            self.ensure_blocks(req, base + ql)
            bt = self.block_tables[req]
            slots.extend(bt[(base + t) // bs] * bs + (base + t) % bs for t in range(ql))
            q_parts.append(q_new)
            k_parts.append(k_new)
            v_parts.append(v_new)
            kv_lens.append(base + ql)

        q_lens = list(reqs.values())
        cu = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32)
        table_cols = max(len(bt) for bt in self.block_tables.values())
        bt_tensor = torch.zeros(len(reqs), table_cols, dtype=torch.int32)
        for i, req in enumerate(reqs):
            row = self.block_tables[req]
            bt_tensor[i, :len(row)] = torch.tensor(row, dtype=torch.int32)

        meta = types.SimpleNamespace(
            attn_state=state,
            causal=True,
            num_actual_tokens=int(cu[-1]),
            query_start_loc=cu,
            seq_lens=torch.tensor(kv_lens, dtype=torch.int32),
            seq_lens_cpu=torch.tensor(kv_lens, dtype=torch.int32),
            seq_lens_list=list(kv_lens),
            actual_seq_lengths_q=cu[1:].tolist(),
            block_tables=bt_tensor,
            slot_mapping=torch.tensor(slots, dtype=torch.long),
            max_query_len=max(q_lens),
            num_decodes=sum(1 for x in q_lens if x <= 4),
            num_prefills=sum(1 for x in q_lens if x > 4),
            num_decode_tokens=sum(x for x in q_lens if x <= 4),
        )
        query = torch.cat(q_parts)
        key = torch.cat(k_parts)
        value = torch.cat(v_parts)
        output = torch.zeros(meta.num_actual_tokens, self.nq * d, dtype=torch.bfloat16)

        self.impl.reshape_and_cache(query, key, value, self.cache, meta, output)
        self.impl.forward_impl(query, key, value, self.cache, meta, output)

        ok = self._check_l1(name) and self._check_l2(name, reqs, query, output)
        return check(f"{name} [{state_name(self.qfa, state)}]", ok)

    def _check_l1(self, name):
        ref = pack_reference(self.golden, self.ledger, self.block_tables,
                             self.nb, self.bs, self.nkv, self.d)
        p = self.impl._planes
        ok = True
        for req, (k_hist, _) in self.ledger.items():
            s = k_hist.shape[0]
            for j, blk in enumerate(self.block_tables[req]):
                valid = max(0, min(self.bs, s - j * self.bs))
                if valid == 0:
                    continue
                vrows = (valid + WINDOW - 1) // WINDOW
                for label, got, want, rows in (
                    ("K", p.k_fp8, ref["k"], valid), ("Ks", p.k_scale, ref["ks"], valid),
                    ("V", p.v_fp8, ref["v"], valid), ("Vs", p.v_scale, ref["vs"], vrows),
                ):
                    if not torch.equal(got[blk][:rows], want[blk][:rows]):
                        diff = (got[blk][:rows] != want[blk][:rows]).sum().item()
                        print(f"    L1 {name}: req {req} block {blk} {label}: {diff} bytes differ")
                        ok = False
        return ok

    def _check_l2(self, name, reqs, query, output):
        # Distribution-sensitive criterion: MXFP8 noise on short-prefix rows
        # (a first token attends to one KV entry, so softmax averages nothing)
        # passes, while any misindexing bug craters cos and the pass rate.
        scale = self.impl.scale
        ok = True
        lo = 0
        for req, ql in reqs.items():
            k_hist, v_hist = self.ledger[req]
            ref = ref_attention(query[lo:lo + ql], k_hist, v_hist, scale)
            got = output[lo:lo + ql].float().reshape(ql, self.nq, self.d)
            diff = (got - ref).abs()
            cos = torch.nn.functional.cosine_similarity(
                got.reshape(-1), ref.reshape(-1), dim=0).item()
            mean_abs = diff.mean().item()
            pass_rate = ((diff <= 0.05) | (diff / ref.abs().clamp(min=1.0) <= 0.1)) \
                .float().mean().item()
            if cos < 0.99 or mean_abs > 0.02 or pass_rate < 0.995:
                print(f"    L2 {name}: req {req} cos={cos:.4f} "
                      f"mean_abs={mean_abs:.5f} pass_rate={pass_rate:.4f}")
                ok = False
            lo += ql
        return ok


def state_name(qfa, state):
    st = qfa.AscendAttentionState
    return {st.PrefillNoCache: "PrefillNoCache", st.DecodeOnly: "DecodeOnly",
            st.ChunkedPrefill: "ChunkedPrefill", st.SpecDecoding: "SpecDecoding"}.get(state, str(state))


def main() -> int:
    torch.manual_seed(20260830)
    qfa = load_qfa_module()
    golden = load_golden_module()
    nq, nkv, d, bs = 24, 4, 256, 128
    sim = EngineSim(qfa, golden, nq, nkv, d, bs, nb=8)
    torch.ops._C_ascend = FakeQfaOps(golden, nq, nkv, d, bs)
    st = qfa.AscendAttentionState

    all_ok = True
    all_ok &= sim.step("S1 solo prefill A(150)", {"A": 150}, st.PrefillNoCache)
    all_ok &= sim.step("S2 decode A(+1)", {"A": 1}, st.DecodeOnly)
    all_ok &= sim.step("S3 mixed A(+1) + prefill B(90)", {"A": 1, "B": 90}, st.ChunkedPrefill)
    all_ok &= sim.step("S4 decode A,B(+1)", {"A": 1, "B": 1}, st.DecodeOnly)
    all_ok &= sim.step("S5 MTP verify A,B(+4)", {"A": 4, "B": 4}, st.SpecDecoding)
    # MTP rejection: A keeps 2 of 4 drafts, B keeps all 4
    sim.rollback("A", sim.ledger["A"][0].shape[0] - 2)
    all_ok &= sim.step("S6 post-reject decode A,B(+1)", {"A": 1, "B": 1}, st.DecodeOnly)
    all_ok &= sim.step("S7 decode A,B(+1)", {"A": 1, "B": 1}, st.DecodeOnly)
    all_ok &= sim.step("S8 verify again A,B(+4)", {"A": 4, "B": 4}, st.SpecDecoding)

    print(f"[{'GREEN' if all_ok else 'RED'}] QFA M1 end-to-end engine-schedule "
          f"simulation (worktree {os.environ['QFA_WORKTREE']})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
