#!/usr/bin/env python3
"""C0 experiment: QFA main op inside torch.npu.graph + per-step AICPU metadata.

Decides the milestone-C aclgraph strategy. Steady-state decode timing under
FULL_DECODE_ONLY is:  refresh persistent buffers -> AICPU metadata (outside
graph) -> D2D copy into a resident buffer -> graph replay. The B-stage 507018
crash came from an AICPU task queued mid-forward right before draft-graph
replays; DSA already ships per-step builder-stage AICPU + FULL replay, so this
experiment must (a) prove QFA's EXEC_NPU_CMD internals survive graph capture
and (b) stress the AICPU/replay adjacency at decode cadence.

  stage 0  capture feasibility: one graph holding the real QFA main op
           (PA_BBND, q_len=4 verify shape) reading resident metadata/seqused/
           block_table/cu_seqlens buffers; replay twice vs eager, bit-compare
  stage 1  correctness across sequence growth: step committed by +1..+4
           (accept-count emulation incl. rollback overwrites), rewrite KV,
           AICPU metadata -> copy_ -> replay, compare vs eager QFA every step
  stage 2  adjacency stress xN (default 2000, QFA_ILV_ITERS to raise):
           (A) metadata->copy->replay        (DSA-like, single graph)
           (C) metadata->copy->replay x2     (target+draft double graph)
           optional (D) QFA_ILV_FIA=1 adds an FIA graph interleave

Verdict guide printed at the end:
  stage0+1+2 all GREEN -> per-step AICPU outside graph is the C3 mainline
  stage2 RED only      -> host-side SectionStreamK port (fallback) required
  stage0 RED           -> QFA needs out-variant/explicit workspace (csrc)

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
    results = {}
    try:
        results["stage0"], graph, out = stage0(h)
    except Exception:
        traceback.print_exc()
        results["stage0"] = False
        graph = out = None
    print()
    if results["stage0"]:
        try:
            results["stage1"] = stage1(h, graph, out, steps)
        except Exception:
            traceback.print_exc()
            results["stage1"] = False
        print()
        try:
            results["stage2"] = stage2(h, graph, iters)
        except Exception:
            traceback.print_exc()
            results["stage2"] = False
        print()

    for name, ok in results.items():
        print(f"[{'GREEN' if ok else 'RED'}] {name}")
    all_ok = all(results.values()) and len(results) == 3
    if all_ok:
        print("[VERDICT] per-step AICPU metadata outside the graph is SAFE -> C3 mainline")
    elif not results.get("stage0", False):
        print("[VERDICT] QFA capture failed -> needs out-variant/explicit workspace (csrc)")
    elif not results.get("stage2", True):
        print("[VERDICT] adjacency unstable -> host-side SectionStreamK port (fallback)")
    print(f"[{'GREEN' if all_ok else 'RED'}] QFA graph interleave overall")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
