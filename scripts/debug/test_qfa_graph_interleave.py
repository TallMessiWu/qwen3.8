#!/usr/bin/env python3
"""C0 experiment: can the whole QFA decode path live inside torch.npu.graph?

Decides the milestone-C3 aclgraph strategy. The target steady-state decode
shape under FULL_DECODE_ONLY is:  refresh resident buffers -> AICPU metadata
(outside the graph) -> D2D copy into a resident buffer -> graph replay. The
B-stage 507018 crash came from an AICPU task queued mid-forward right before
draft-graph replays, so this has to prove three separate things: that QFA's
EXEC_NPU_CMD internals survive capture, that the *write* path's operators do
too, and that AICPU next to a replay is stable at decode cadence.

  stage 0  capture feasibility: one graph holding the real QFA main op
           (PA_BBND, q_len=4 verify shape) reading resident metadata/seqused/
           block_table/cu_seqlens buffers; replay twice vs eager, bit-compare
  stage 1  correctness across sequence growth: step committed by +1..+4
           (accept-count emulation incl. rollback overwrites), rewrite KV,
           AICPU metadata -> copy_ -> replay, compare vs eager QFA every step
  stage 3  the WRITE path under capture: npu_dynamic_mx_quant, index_put_ with
           device indices, searchsorted, gather, exp2/repeat_interleave/clamp.
           stage 0 only ever captured the read; nothing proved these survive.
  stage 4  the two-write restore inside one graph: confirmed-scale write ->
           main op -> unclamped restore write. Also asserts the invariant the
           restore exists for - the committed cache ends up byte-identical to
           what a single unclamped write leaves.
  stage 5  resident buffers whose CONTENTS change: the block table is
           rewritten between replays. A graph that baked the capture-time
           table passes stages 0-4 and fails here.
  stage 2  adjacency stress xN (default 2000, QFA_ILV_ITERS to raise), run
           last because it is the slow one:
           (A) metadata->copy->replay        (DSA-like, single graph)
           (C) metadata->copy->replay x2     (target+draft double graph)
           optional (D) QFA_ILV_FIA=1 adds an FIA graph interleave

Verdict guide printed at the end:
  all GREEN      -> per-step AICPU outside the graph is the C3 mainline
  stage2 RED     -> host-side SectionStreamK port (fallback) required
  stage3/4 RED   -> the write path has to stay outside the captured region
  stage5 RED     -> the graph is not reading the resident buffers
  stage0 RED     -> QFA needs out-variant/explicit workspace (csrc)

Stages 3-5 re-implement the impl's write arithmetic rather than importing it:
the real ones are methods on an attention impl that needs a live vLLM config.
plan_v_windows IS imported, so the indexing that bit us before is the real
code; the quantize/clamp/dequant helpers below are copies of attention_v1.py
and have to be kept in step with it.

Run: python scripts/debug/test_qfa_graph_interleave.py   (1 idle NPU)
"""

import math
import os
import sys
import traceback

import torch

GROUP = 32
BS = 128
FP8 = None

B = 2
QLEN = 4  # 1 + num_spec(3): MTP verify shape
NQ, NKV, D = 24, 4, 256  # 27B per-rank
G = NQ // NKV
MAX_LEN = 1024
BN_PER_SEQ = MAX_LEN // BS
SCALE = 1.0 / math.sqrt(D)


def quant_qk(x: torch.Tensor):
    """(T,N,D) bf16 -> fp8 (T,N,D) + e8m0-byte scale (T,N,D//64,2). NPU."""
    import torch_npu

    t, n, d = x.shape
    fp8, scale = torch_npu.npu_dynamic_mx_quant(x.reshape(t * n, d), dst_type=FP8, scale_alg=0)
    return fp8.reshape(t, n, d), scale.view(torch.uint8).reshape(t, n, d // 64, 2)


def quant_cols64(rows: torch.Tensor):
    """(64,N,D) bf16 -> fp8 (64,N,D) + v-scale bytes (N,D,2). NPU."""
    import torch_npu

    n, d = rows.shape[1], rows.shape[2]
    cols = rows.permute(1, 2, 0).reshape(n * d, 64).contiguous()
    fp8, scale = torch_npu.npu_dynamic_mx_quant(cols, dst_type=FP8, scale_alg=0)
    fp8 = fp8.reshape(n, d, 64).permute(2, 0, 1).contiguous()
    return fp8, scale.view(torch.uint8).reshape(n, d, 2)


class Harness:
    """Persistent buffers + KV rewrite helpers shared by all stages."""

    def __init__(self):
        bn_total = B * BN_PER_SEQ
        dev = torch.device("npu")
        self.k_fp8 = torch.zeros(bn_total, BS, NKV, D, dtype=torch.uint8, device=dev)
        self.v_fp8 = torch.zeros(bn_total, BS, NKV, D, dtype=torch.uint8, device=dev)
        self.k_scale = torch.zeros(bn_total, BS, NKV, D // 64, 2, dtype=torch.uint8, device=dev)
        self.v_scale = torch.zeros(bn_total, BS // 64, NKV, D, 2, dtype=torch.uint8, device=dev)
        self.block_table = (
            torch.arange(bn_total, dtype=torch.int32, device=dev).reshape(B, BN_PER_SEQ)
        ).contiguous()
        self.seqused = torch.zeros(B, dtype=torch.int32, device=dev)
        self.cu_q = torch.tensor(
            [i * QLEN for i in range(B + 1)], dtype=torch.int32, device=dev
        )
        self.q_fp8 = torch.zeros(B * QLEN, NQ, D, dtype=torch.uint8, device=dev)
        self.q_descale = torch.zeros(NKV, B * QLEN, G, D // 64, 2, dtype=torch.uint8, device=dev)
        self.metadata_buf = torch.zeros(4096, dtype=torch.int32, device=dev)
        self.mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()
        # deterministic source tokens per request
        torch.manual_seed(2027)
        self.src_k = torch.randn(B, MAX_LEN, NKV, D, dtype=torch.bfloat16, device=dev)
        self.src_v = torch.randn(B, MAX_LEN, NKV, D, dtype=torch.bfloat16, device=dev)
        self.src_q = torch.randn(B, MAX_LEN, NQ, D, dtype=torch.bfloat16, device=dev)

    def prefill_all(self) -> None:
        """Populate the whole cache once so every read window holds real data."""
        for b in range(B):
            base_slot = b * BN_PER_SEQ * BS
            kf, ks = quant_qk(self.src_k[b])
            slots = torch.arange(MAX_LEN, device="npu") + base_slot
            self.k_fp8.view(-1, NKV, D)[slots] = kf.view(torch.uint8)
            self.k_scale.view(-1, NKV, D // 64, 2)[slots] = ks
            for w in range(MAX_LEN // 64):
                vf, vsc = quant_cols64(self.src_v[b, w * 64 : (w + 1) * 64])
                wslot = base_slot // 64 + w
                self.v_fp8.view(-1, 64, NKV, D)[wslot] = vf.view(torch.uint8)
                self.v_scale.view(-1, NKV, D, 2)[wslot] = vsc
        torch.npu.synchronize()

    def common_args(self, max_kv: int) -> dict:
        return {
            "cu_seqlens_q": self.cu_q,
            "seqused_kv": self.seqused,
            "mask_mode": 3,
            "max_seqlen_q": QLEN,
            "max_seqlen_kv": max_kv,
            "layout_q": "TND",
            "layout_q_descale": "N2TGD",
            "layout_kv": "PA_BBND",
            "layout_out": "TND",
        }

    def launch_metadata(self, max_kv: int) -> torch.Tensor:
        vs = self.v_scale.view(torch.float8_e8m0fnu)
        return torch.ops._C_ascend.npu_quant_flash_attn_metadata(
            NQ, NKV, D, v_descale=vs, **self.common_args(max_kv)
        )

    def qfa(self, metadata: torch.Tensor, max_kv: int) -> torch.Tensor:
        return torch.ops._C_ascend.npu_quant_flash_attn(
            self.q_fp8.view(FP8),
            self.k_fp8.view(FP8),
            self.v_fp8.view(FP8),
            self.q_descale.view(torch.float8_e8m0fnu),
            self.k_scale.view(torch.float8_e8m0fnu),
            self.v_scale.view(torch.float8_e8m0fnu),
            metadata,
            SCALE,
            block_table=self.block_table,
            attn_mask=self.mask,
            **self.common_args(max_kv),
        )

    def write_step(self, committed: int) -> None:
        """Write tokens [committed, committed+QLEN) of every request into the
        caches (K per-token, V whole-window rewrite) and refresh q buffers."""
        new_end = committed + QLEN
        for b in range(B):
            base_slot = b * BN_PER_SEQ * BS
            kf, ks = quant_qk(self.src_k[b, committed:new_end])
            flat_k = self.k_fp8.view(-1, NKV, D)
            flat_ks = self.k_scale.view(-1, NKV, D // 64, 2)
            slots = torch.arange(committed, new_end, device="npu") + base_slot
            flat_k[slots] = kf.view(torch.uint8)
            flat_ks[slots] = ks
            w_last = (new_end - 1) // 64
            for w in {max(0, w_last - 1), w_last}:
                valid = min(64, new_end - w * 64)
                rows = torch.zeros(64, NKV, D, dtype=torch.bfloat16, device="npu")
                rows[:valid] = self.src_v[b, w * 64 : w * 64 + valid]
                vf, vsc = quant_cols64(rows)
                wslot = base_slot // 64 + w
                self.v_fp8.view(-1, 64, NKV, D)[wslot] = vf.view(torch.uint8)
                self.v_scale.view(-1, NKV, D, 2)[wslot] = vsc
            qf, qs = quant_qk(self.src_q[b, committed:new_end])
            self.q_fp8[b * QLEN : (b + 1) * QLEN] = qf.view(torch.uint8)
            qs_n2tgd = qs.reshape(QLEN, NKV, G, D // 64, 2).permute(1, 0, 2, 3, 4)
            self.q_descale[:, b * QLEN : (b + 1) * QLEN] = qs_n2tgd
        self.seqused.fill_(new_end)


def stage0(h: Harness) -> tuple[bool, torch.npu.NPUGraph, torch.Tensor]:
    print("== stage 0: capture the real QFA main op ==")
    h.prefill_all()
    committed = 60  # crosses a 64-boundary with QLEN=4
    h.write_step(committed)
    md = h.launch_metadata(MAX_LEN)
    h.metadata_buf.copy_(md)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = h.qfa(h.metadata_buf, MAX_LEN)
    torch.npu.synchronize()
    print(f"  [s0] capture OK, out shape={tuple(out.shape)} dtype={out.dtype}", flush=True)

    ok = True
    for trial, base in enumerate([60, 124]):  # second one crosses 128 block edge
        h.write_step(base)
        h.metadata_buf.copy_(h.launch_metadata(MAX_LEN))
        graph.replay()
        torch.npu.synchronize()
        ref = h.qfa(h.launch_metadata(MAX_LEN), MAX_LEN)
        torch.npu.synchronize()
        same = torch.equal(out.cpu(), ref.cpu())
        finite = bool(torch.isfinite(out.float()).all())
        print(f"  [s0] replay#{trial} committed={base} equal-vs-eager={same} finite={finite}")
        ok = ok and same and finite
    print(f"  [s0] {'GREEN' if ok else 'RED'}")
    return ok, graph, out


def stage1(h: Harness, graph: torch.npu.NPUGraph, out: torch.Tensor, steps: int) -> bool:
    print(f"== stage 1: correctness over {steps} growth steps (+1..+4, rollbacks) ==")
    torch.manual_seed(3)
    committed, ok, checked = 8, True, 0
    for step in range(steps):
        if committed + QLEN >= MAX_LEN - QLEN:
            break
        h.write_step(committed)
        h.metadata_buf.copy_(h.launch_metadata(MAX_LEN))
        graph.replay()
        ref = h.qfa(h.launch_metadata(MAX_LEN), MAX_LEN)
        torch.npu.synchronize()
        if not torch.equal(out.cpu(), ref.cpu()):
            print(f"  [s1] step {step} committed={committed}: replay != eager")
            ok = False
            break
        checked += 1
        committed += int(torch.randint(1, QLEN + 1, (1,)))  # accept 1..4 => rollback overwrites
    print(f"  [s1] {checked} steps verified, final committed={committed}")
    print(f"  [s1] {'GREEN' if ok else 'RED'}")
    return ok


def stage2(h: Harness, graph: torch.npu.NPUGraph, iters: int) -> bool:
    print(f"== stage 2: AICPU/replay adjacency stress x{iters} ==")
    h.write_step(200)
    torch.npu.synchronize()

    ok = True
    try:
        for i in range(iters):  # (A) metadata -> copy -> replay
            h.metadata_buf.copy_(h.launch_metadata(MAX_LEN))
            graph.replay()
            if (i + 1) % 500 == 0:
                torch.npu.synchronize()
                print(f"  [s2-A] {i + 1}/{iters} alive", flush=True)
        torch.npu.synchronize()
        print("  [s2-A] GREEN (single graph)")
    except Exception:
        traceback.print_exc()
        print("  [s2-A] RED — per-step AICPU + replay is NOT safe")
        return False

    graph2 = torch.npu.NPUGraph()
    with torch.npu.graph(graph2):
        out2 = h.qfa(h.metadata_buf, MAX_LEN)  # noqa: F841 - keep graph output alive
    torch.npu.synchronize()
    try:
        for i in range(iters):  # (C) metadata -> copy -> replay x2 (target+draft)
            h.metadata_buf.copy_(h.launch_metadata(MAX_LEN))
            graph.replay()
            graph2.replay()
            if (i + 1) % 500 == 0:
                torch.npu.synchronize()
                print(f"  [s2-C] {i + 1}/{iters} alive", flush=True)
        torch.npu.synchronize()
        print("  [s2-C] GREEN (double graph)")
    except Exception:
        traceback.print_exc()
        print("  [s2-C] RED — AICPU adjacent to double replay crashes")
        ok = False

    if os.environ.get("QFA_ILV_FIA", "0") == "1" and ok:
        import torch_npu

        q_bf = torch.randn(B, NQ, 1, D, dtype=torch.bfloat16, device="npu")
        graph3 = torch.npu.NPUGraph()
        with torch.npu.graph(graph3):
            fia_out = torch_npu.npu_fused_infer_attention_score(  # noqa: F841
                q_bf, q_bf.expand(B, NQ, 1, D).contiguous(), q_bf.contiguous(),
                num_heads=NQ, input_layout="BNSD", scale=SCALE,
            )
        torch.npu.synchronize()
        try:
            for i in range(iters):  # (D) mixed FIA graph in the cadence
                h.metadata_buf.copy_(h.launch_metadata(MAX_LEN))
                graph.replay()
                graph3.replay()
                if (i + 1) % 500 == 0:
                    torch.npu.synchronize()
                    print(f"  [s2-D] {i + 1}/{iters} alive", flush=True)
            torch.npu.synchronize()
            print("  [s2-D] GREEN (FIA interleave)")
        except Exception:
            traceback.print_exc()
            print("  [s2-D] RED")
            ok = False
    print(f"  [s2] {'GREEN' if ok else 'RED'}")
    return ok


# --------------------------------------------------------------------------
# The write path, mirrored from AscendAttentionBackendImpl. Keep in step with
# attention_v1.py: a copy that has drifted tests nothing.
# --------------------------------------------------------------------------

VGROUP = 64  # tokens per V window - two packed 32-token E8M0 rows
E4M3_MAX = 448.0
PLANES = ("k_fp8", "k_scale", "v_fp8", "v_scale")
_STEP_SRC: dict[str, torch.Tensor] = {}


def _planes(h):
    return (h.k_fp8, h.k_scale, h.v_fp8, h.v_scale)


def snapshot(h):
    return tuple(t.clone() for t in _planes(h))


def restore(h, snap) -> None:
    for dst, src in zip(_planes(h), snap):
        dst.copy_(src)


def _quant_along_tokens(rows: torch.Tensor):
    """(W,64,N,D) bf16 -> fp8 bytes (W,64,N,D) + scale bytes (W,N,D,2)."""
    import torch_npu

    w, group, n, d = rows.shape
    cols = rows.permute(0, 2, 3, 1).reshape(w * n * d, group)
    fp8, scale = torch_npu.npu_dynamic_mx_quant(cols.contiguous(), dst_type=FP8, scale_alg=0)
    fp8 = fp8.view(torch.uint8).reshape(w, n, d, group).permute(0, 3, 1, 2)
    scale = scale.view(torch.uint8).reshape(w, n, d, 2)
    return fp8.contiguous(), scale


def _clamp_to_scale(rows: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Hold (W,64,N,D) inside what e4m3 represents at these E8M0 bytes."""
    limit = (torch.exp2((scale.to(torch.int32) - 127).to(torch.float32)) * E4M3_MAX).to(rows.dtype)
    limit = torch.where(scale == 0, rows.new_full((), torch.finfo(rows.dtype).max), limit)
    limit = limit.permute(0, 3, 1, 2).repeat_interleave(VGROUP // 2, dim=1)
    return torch.clamp(rows, -limit, limit)


def _dequant_windows(h, slots: torch.Tensor, n: int, d: int) -> torch.Tensor:
    fp8 = h.v_fp8.view(-1, VGROUP, n, d)[slots]
    scale = h.v_scale.view(-1, n, d, 2)[slots]
    vals = fp8.view(FP8).to(torch.bfloat16)
    factor = torch.exp2((scale.to(torch.int32) - 127).to(torch.float32)).to(torch.bfloat16)
    factor = factor.permute(0, 3, 1, 2).repeat_interleave(VGROUP // 2, dim=1)
    return vals * factor


def _commit_windows(h, rows: torch.Tensor, slots: torch.Tensor, n: int, d: int) -> None:
    fp8, scale = _quant_along_tokens(rows)
    h.v_fp8.view(-1, VGROUP, n, d).index_put_((slots,), fp8)
    h.v_scale.view(-1, n, d, 2).index_put_((slots,), scale)


def _window_keep(valid: torch.Tensor, group: int) -> torch.Tensor:
    return torch.arange(group, device=valid.device).unsqueeze(0) < valid.unsqueeze(1)


def write_kv_step(h, key, value, slot_mapping, qsl, seq_lens, block_tables, num_reqs, spec_verify):
    """Mirror of the decode branch of _qfa_write_kv. Returns the restore payload."""
    import torch_npu
    from vllm_ascend.attention.qfa_scale import plan_v_windows

    num_tokens, n, d = key.shape
    fp8, scale = torch_npu.npu_dynamic_mx_quant(key.reshape(num_tokens * n, d), dst_type=FP8, scale_alg=0)
    fp8 = fp8.reshape(num_tokens, n, d).view(torch.uint8)
    scale = scale.view(torch.uint8).reshape(num_tokens, n, d // 64, 2)
    safe = torch.where(slot_mapping >= 0, slot_mapping, torch.zeros_like(slot_mapping)).to(torch.int64)
    h.k_fp8.view(-1, n, d).index_put_((safe,), fp8)
    h.k_scale.view(-1, n, d // 64, 2).index_put_((safe,), scale)

    token_ids = torch.arange(num_tokens, device=value.device)
    starts = qsl[:num_reqs]
    req_ids = torch.searchsorted(qsl[1 : num_reqs + 1].contiguous(), token_ids, right=True)
    ctx_lens = seq_lens[:num_reqs] - (qsl[1 : num_reqs + 1] - starts)
    positions = ctx_lens[req_ids] + (token_ids - starts[req_ids])

    plan = plan_v_windows(ctx_lens, seq_lens, block_tables, num_reqs, VGROUP)
    slots = plan["window_slots"]
    raw_rows = _dequant_windows(h, slots, n, d)
    token_window = torch.div(positions, VGROUP, rounding_mode="floor")
    scratch = raw_rows.new_zeros(1, VGROUP, n, d)
    rows_ext = torch.cat([raw_rows, scratch])
    discard = raw_rows.shape[0]
    for pair_slot, window_of_row in ((0, plan["first_window"]), (1, plan["last_window"])):
        target = torch.where(
            token_window == window_of_row[req_ids],
            req_ids * 2 + pair_slot,
            torch.full_like(req_ids, discard),
        )
        rows_ext[target, positions % VGROUP] = value
    raw_rows = rows_ext[:discard]
    rows = raw_rows * _window_keep(plan["valid"], VGROUP).view(-1, VGROUP, 1, 1)

    if not spec_verify:
        _commit_windows(h, rows, slots, n, d)
        return None
    _, scale_read = _quant_along_tokens(rows * _window_keep(plan["confirmed"], VGROUP).view(-1, VGROUP, 1, 1))
    _commit_windows(h, _clamp_to_scale(rows, scale_read), slots, n, d)
    return (rows, slots, n, d)


def decode_inputs(h, committed: int, num_reqs: int = B):
    """The device tensors a verify step hands the write path.

    The K/V it feeds are deliberately NOT the source prefill_all committed, so
    a write that silently does nothing cannot pass as a byte-for-byte match.
    """
    if not _STEP_SRC:
        torch.manual_seed(4243)
        _STEP_SRC["k"] = torch.randn(B, MAX_LEN, NKV, D, dtype=torch.bfloat16, device="npu")
        _STEP_SRC["v"] = torch.randn(B, MAX_LEN, NKV, D, dtype=torch.bfloat16, device="npu")
    src_k, src_v = _STEP_SRC["k"], _STEP_SRC["v"]
    dev = torch.device("npu")
    end = committed + QLEN
    qsl = torch.tensor([i * QLEN for i in range(num_reqs + 1)], dtype=torch.int32, device=dev)
    seq_lens = torch.full((num_reqs,), end, dtype=torch.int32, device=dev)
    key = torch.stack([src_k[b, committed:end] for b in range(num_reqs)]).reshape(-1, NKV, D).contiguous()
    value = torch.stack([src_v[b, committed:end] for b in range(num_reqs)]).reshape(-1, NKV, D).contiguous()
    offsets = [torch.arange(committed, end, device=dev) + b * BN_PER_SEQ * BS for b in range(num_reqs)]
    slots = torch.cat(offsets).to(torch.int64)
    return key, value, slots, qsl, seq_lens


def stage3(h) -> bool:
    print("== stage 3: the write path under capture ==")
    committed = 62  # the 4 new tokens straddle a 64-token V window boundary
    h.write_step(committed)  # realistic q buffers / seqused for this length
    torch.npu.synchronize()
    key, value, slots, qsl, seq_lens = decode_inputs(h, committed)
    base = snapshot(h)

    write_kv_step(h, key, value, slots, qsl, seq_lens, h.block_table, B, spec_verify=False)
    torch.npu.synchronize()
    eager = snapshot(h)

    restore(h, base)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        write_kv_step(h, key, value, slots, qsl, seq_lens, h.block_table, B, spec_verify=False)
    torch.npu.synchronize()
    print("  [s3] capture OK")
    restore(h, base)
    graph.replay()
    torch.npu.synchronize()
    replayed = snapshot(h)

    ok = True
    for name, before, e, r in zip(PLANES, base, eager, replayed):
        same = torch.equal(e.cpu(), r.cpu())
        # Without this the check passes when both runs write nothing at all.
        touched = int((e != before).sum())
        print(f"  [s3] {name:<8} replay==eager={same} bytes-the-write-changed={touched}")
        ok = ok and same and touched > 0
    print(f"  [s3] {'GREEN' if ok else 'RED'}")
    return ok


def stage4(h) -> bool:
    print("== stage 4: the two-write restore inside one graph ==")
    committed = 126  # straddles both a V window and a 128-token kernel block
    h.write_step(committed)
    torch.npu.synchronize()
    key, value, slots, qsl, seq_lens = decode_inputs(h, committed)
    h.metadata_buf.copy_(h.launch_metadata(MAX_LEN))
    torch.npu.synchronize()
    base = snapshot(h)

    def two_write():
        pending = write_kv_step(h, key, value, slots, qsl, seq_lens, h.block_table, B, spec_verify=True)
        attn = h.qfa(h.metadata_buf, MAX_LEN)
        _commit_windows(h, *pending)
        return attn

    # Where a single unclamped write lands. The restore exists to end up here,
    # so the committed cache has to match it byte for byte.
    write_kv_step(h, key, value, slots, qsl, seq_lens, h.block_table, B, spec_verify=False)
    torch.npu.synchronize()
    single = snapshot(h)

    restore(h, base)
    eager_out = two_write().cpu().clone()
    torch.npu.synchronize()
    eager_cache = snapshot(h)

    restore(h, base)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_out = two_write()
    torch.npu.synchronize()
    print("  [s4] capture OK")
    restore(h, base)
    graph.replay()
    torch.npu.synchronize()

    ok = torch.equal(eager_out, graph_out.cpu())
    print(f"  [s4] attention output replay==eager={ok}")
    for name, e, r, s in zip(PLANES, eager_cache, snapshot(h), single):
        same = torch.equal(e.cpu(), r.cpu())
        restored = torch.equal(e.cpu(), s.cpu())
        print(f"  [s4] {name:<8} replay==eager={same} restore-lands-on-single-write={restored}")
        ok = ok and same and restored
    print(f"  [s4] {'GREEN' if ok else 'RED'}")
    return ok


def stage5(h, graph, out) -> bool:
    print("== stage 5: resident buffers whose contents change between replays ==")
    original = h.block_table.clone()
    # Same cache bytes, different rows: request 0 now reads request 1 blocks.
    swapped = torch.stack([original[1], original[0]]).contiguous()

    def replay_with(table):
        h.block_table.copy_(table)
        h.metadata_buf.copy_(h.launch_metadata(MAX_LEN))
        graph.replay()
        torch.npu.synchronize()
        return out.cpu().clone()

    ok = True
    seen = {}
    for name, table in (("swapped", swapped), ("original", original)):
        got = replay_with(table)
        ref = h.qfa(h.launch_metadata(MAX_LEN), MAX_LEN)
        torch.npu.synchronize()
        same = torch.equal(got, ref.cpu())
        print(f"  [s5] block table {name:<8} replay==eager={same}")
        seen[name] = got
        ok = ok and same
    # A graph that baked in the capture-time table answers both the same way.
    differs = not torch.equal(seen["swapped"], seen["original"])
    print(f"  [s5] swapping the table changes the replay output={differs} (else the check is vacuous)")
    h.block_table.copy_(original)
    ok = ok and differs
    print(f"  [s5] {'GREEN' if ok else 'RED'}")
    return ok


def main() -> int:
    global FP8
    FP8 = torch.float8_e4m3fn
    device_id = int(os.environ.get("QFA_TEST_DEVICE", "0"))
    import torch_npu  # noqa: F401

    torch.npu.set_device(device_id)
    print(f"[INFO] device npu:{device_id}")
    try:
        from vllm_ascend.utils import bootstrap_custom_op_env

        bootstrap_custom_op_env(include_vendor_lib=True)
    except Exception:
        import vllm_ascend

        vendor = os.path.join(
            os.path.dirname(vllm_ascend.__file__), "_cann_ops_custom", "vendors", "custom_transformer"
        )
        if os.path.isdir(vendor):
            prev = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = vendor + (":" + prev if prev else "")
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    iters = int(os.environ.get("QFA_ILV_ITERS", "2000"))
    steps = int(os.environ.get("QFA_ILV_STEPS", "128"))

    h = Harness()
    results: dict[str, bool] = {}
    try:
        results["stage0"], graph, out = stage0(h)
    except Exception:
        traceback.print_exc()
        results["stage0"] = False
        graph = out = None
    print()
    if results["stage0"]:
        # stage 2 is the slow one and runs last, so a cheap stage that fails
        # reports in seconds rather than after 2000 iterations.
        for name, run in (
            ("stage1", lambda: stage1(h, graph, out, steps)),
            ("stage3", lambda: stage3(h)),
            ("stage4", lambda: stage4(h)),
            ("stage5", lambda: stage5(h, graph, out)),
            ("stage2", lambda: stage2(h, graph, iters)),
        ):
            try:
                results[name] = run()
            except Exception:
                traceback.print_exc()
                results[name] = False
            print()

    for name in sorted(results):
        print(f"[{'GREEN' if results[name] else 'RED'}] {name}")
    all_ok = all(results.values()) and len(results) == 6
    if all_ok:
        print("[VERDICT] per-step AICPU metadata outside the graph is SAFE -> C3 mainline")
    elif not results.get("stage0", False):
        print("[VERDICT] QFA capture failed -> needs out-variant/explicit workspace (csrc)")
    elif not (results.get("stage3", True) and results.get("stage4", True)):
        print("[VERDICT] the write path cannot be captured -> keep it outside the captured region")
    elif not results.get("stage5", True):
        print("[VERDICT] the replay is not reading the resident buffers -> check what got baked in")
    elif not results.get("stage2", True):
        print("[VERDICT] adjacency unstable -> host-side SectionStreamK port (fallback)")
    print(f"[{'GREEN' if all_ok else 'RED'}] QFA graph interleave overall")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
