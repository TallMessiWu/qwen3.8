#!/usr/bin/env python3
"""C3 regression triage: pin down WHICH mechanism broke the smoke runs.

After ada6e3fc5..cbfaf9bdd the eager smoke shows three regressions the C0
probe (all GREEN) did not predict:
  (1) NUM_SPEC=3 dies with 507018 in QuantFlashAttnMetadata during the first
      MTP proposal;
  (2) NUM_SPEC=0 runs ~100x slower than the ee45dce8c baseline (first decode
      stalls ~350 s, then ~3.8 s/step);
  (3) NUM_SPEC=0 prompt 1 (batch row 1) diverges hard while rows 0/2
      reproduce the old numbers bit for bit.

The probe used B=2 / q_len=4 / fresh tensors; the engine runs B=3 / q_len=1 /
resident-slot slices, and interleaves TND prefill metadata with PA_BBND decode
metadata. Every stage below reproduces one engine-shaped mechanism in
isolation, so one run says which of them is broken:

  stage L  first-call + steady-state latency of the metadata AICPU op in the
           ENGINE decode shape (B=3, q_len=1). RED if steady-state > 100 ms -
           that alone explains the 3.8 s/step (16 layers share 1 call, but a
           slow call is a slow step). The first-call number, if huge, explains
           the 350 s stall.
  stage T  per-step wall time of four task mixes x30 steps:
             T-old : 16x (metadata + FA)          <- ee45dce8c per-layer form
             T-new : 1x metadata + 16x (blob copy_ + FA)   <- C3 form
             T-fa  : 16x FA only                  <- floor
             T-pin : T-new + the 3 resident-buffer refresh copies build() does
           RED if T-new (or T-pin) >> T-old: the C3 step form itself is slow,
           and the delta names the culprit component.
  stage M  the NUM_SPEC=3 crash shape: TND prefill metadata calls (the target
           prefill) immediately followed by PA_BBND decode metadata calls (the
           draft steps), inputs sliced from oversized resident buffers, x200.
           RED (507018) = crash reproduced with a 300-line recipe.
  stage B  batch-position check for regression (3): one batch [r0, r1, r2]
           with unequal seqused vs three single-request calls; each row must
           match its solo run bitwise. A RED row = the paged op mixes rows
           when fed resident-slice inputs (capacity > batch).

Verdict guide:
  L RED             -> the slowdown is the AICPU op itself in the q_len=1
                       shape; look at SectionStreamK for that shape, not at
                       the vllm-ascend python.
  T RED, L GREEN    -> the slowdown is task adjacency (blob copy_ / pin
                       copies), component named by which mix is slow.
  M RED             -> crash reproduced; iterate on this recipe, not the
                       500 s smoke.
  B RED             -> regression (3) is a real batch bug in the paged read
                       with oversized resident inputs; (1)/(2) likely follow.
  all GREEN         -> none of these mechanisms is broken in isolation; the
                       regressions need engine state - next stop is profiling
                       the smoke itself.

Run: python scripts/debug/diag_qfa_c3_step.py   (1 idle NPU, no weights)
"""

import math
import sys
import time
import traceback

import torch

NQ, NKV, D = 24, 4, 256  # 27B per-rank attention geometry
G = NQ // NKV
BS = 128  # QFA kernel block, matches the probe and the engine planes
B = 3  # engine smoke batch
CAPACITY = 6  # engine resident-buffer capacity (max_num_seqs + 2)
MAX_LEN = 1024
BN_PER_SEQ = MAX_LEN // BS
SCALE = 1.0 / math.sqrt(D)
FP8 = None
E8M0 = None

SEQUSED = [70, 70, 74]  # mid-generation engine shape
PREFILL_LENS = [5, 5, 9]  # the smoke prompts


def _quant(x2d: torch.Tensor):
    import torch_npu

    return torch_npu.npu_dynamic_mx_quant(x2d, dst_type=FP8, scale_alg=0)


def quant_tnd(x: torch.Tensor):
    """(T,N,D) bf16 -> fp8 + packed (T,N,D//64,2) e8m0 bytes."""
    t, n, d = x.shape
    fp8, scale = _quant(x.reshape(t * n, d))
    return fp8.reshape(t, n, d), scale.view(torch.uint8).reshape(t, n, d // 64, 2)


class Rig:
    """Engine-shaped buffers: oversized residents, [:B] slices handed out."""

    def __init__(self):
        dev = torch.device("npu")
        torch.manual_seed(2027)
        bn_total = B * BN_PER_SEQ
        self.k_fp8 = torch.zeros(bn_total, BS, NKV, D, dtype=torch.uint8, device=dev)
        self.v_fp8 = torch.zeros(bn_total, BS, NKV, D, dtype=torch.uint8, device=dev)
        self.k_scale = torch.zeros(bn_total, BS, NKV, D // 64, 2, dtype=torch.uint8, device=dev)
        self.v_scale = torch.zeros(bn_total, BS // 64, NKV, D, 2, dtype=torch.uint8, device=dev)
        # Resident buffers the engine way: CAPACITY rows, only [:B] written.
        # The tail rows stay whatever torch.empty found - that is the point.
        self.res_cu = torch.empty(CAPACITY + 1, dtype=torch.int32, device=dev)
        self.res_seqused = torch.empty(CAPACITY, dtype=torch.int32, device=dev)
        self.res_bt = torch.empty(CAPACITY, BN_PER_SEQ, dtype=torch.int32, device=dev)
        self.res_cu[: B + 1] = torch.arange(B + 1, dtype=torch.int32, device=dev)
        self.res_seqused[:B] = torch.tensor(SEQUSED, dtype=torch.int32, device=dev)
        self.res_bt[:B] = torch.arange(bn_total, dtype=torch.int32, device=dev).reshape(B, BN_PER_SEQ)
        self.metadata_buf = torch.zeros(4096, dtype=torch.int32, device=dev)
        self.mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()
        # Populate enough KV that every SEQUSED window reads real data.
        src_k = torch.randn(B, MAX_LEN, NKV, D, dtype=torch.bfloat16, device=dev)
        src_v = torch.randn(B, MAX_LEN, NKV, D, dtype=torch.bfloat16, device=dev)
        for b in range(B):
            base_slot = b * BN_PER_SEQ * BS
            kf, ks = quant_tnd(src_k[b])
            slots = torch.arange(MAX_LEN, device=dev) + base_slot
            self.k_fp8.view(-1, NKV, D)[slots] = kf.view(torch.uint8)
            self.k_scale.view(-1, NKV, D // 64, 2)[slots] = ks
            for w in range(MAX_LEN // 64):
                rows = src_v[b, w * 64 : (w + 1) * 64]
                cols = rows.permute(1, 2, 0).reshape(NKV * D, 64).contiguous()
                vf, vs = _quant(cols)
                wslot = base_slot // 64 + w
                self.v_fp8.view(-1, 64, NKV, D)[wslot] = (
                    vf.view(torch.uint8).reshape(NKV, D, 64).permute(2, 0, 1).contiguous()
                )
                self.v_scale.view(-1, NKV, D, 2)[wslot] = vs.view(torch.uint8).reshape(NKV, D, 2)
        # q_len=1 decode query for the whole batch
        q = torch.randn(B, NQ, D, dtype=torch.bfloat16, device=dev)
        qf, qs = quant_tnd(q)
        self.q_fp8 = qf.view(torch.uint8)
        self.q_descale = (
            qs.reshape(B, NKV, G, D // 64, 2).permute(1, 0, 2, 3, 4).contiguous()
        )
        torch.npu.synchronize()

    def decode_args(self, rows: int = B) -> dict:
        return {
            "cu_seqlens_q": self.res_cu[: rows + 1],
            "seqused_kv": self.res_seqused[:rows],
            "mask_mode": 3,
            "max_seqlen_q": 1,  # engine: _qfa_decode_threshold with NUM_SPEC=0
            "max_seqlen_kv": max(SEQUSED[:rows]),
            "layout_q": "TND",
            "layout_q_descale": "N2TGD",
            "layout_kv": "PA_BBND",
            "layout_out": "TND",
        }

    def launch_metadata(self, rows: int = B) -> torch.Tensor:
        return torch.ops._C_ascend.npu_quant_flash_attn_metadata(
            NQ, NKV, D, v_descale=self.v_scale.view(E8M0), **self.decode_args(rows)
        )

    def launch_fa(self, metadata: torch.Tensor, rows: int = B) -> torch.Tensor:
        return torch.ops._C_ascend.npu_quant_flash_attn(
            self.q_fp8[:rows].view(FP8),
            self.k_fp8.view(FP8),
            self.v_fp8.view(FP8),
            self.q_descale[:, :rows].contiguous().view(E8M0),
            self.k_scale.view(E8M0),
            self.v_scale.view(E8M0),
            metadata,
            SCALE,
            block_table=self.res_bt[:rows],
            attn_mask=self.mask,
            **self.decode_args(rows),
        )


def timed(fn, sync_first: bool = True) -> float:
    if sync_first:
        torch.npu.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.npu.synchronize()
    return time.perf_counter() - t0


def stage_l(rig: Rig) -> bool:
    print("== stage L: metadata AICPU latency, engine decode shape ==")
    first = timed(lambda: rig.launch_metadata())
    print(f"  [L] first call (includes so load): {first * 1e3:.1f} ms")
    steady = timed(lambda: [rig.launch_metadata() for _ in range(20)]) / 20
    print(f"  [L] steady state over 20 calls:    {steady * 1e3:.1f} ms/call")
    ok = steady < 0.1
    print(f"  [L] {'GREEN' if ok else 'RED'} (gate: steady < 100 ms/call)")
    if first > 5.0:
        print(f"  [L] NOTE: first call took {first:.1f} s - that is the smoke's startup stall")
    return ok


def stage_t(rig: Rig) -> bool:
    print("== stage T: per-step wall time of the four task mixes ==")
    steps, layers = 30, 16
    pinned_cu = torch.arange(B + 1, dtype=torch.int32).pin_memory()
    plain_seq = torch.tensor(SEQUSED, dtype=torch.int32)  # NOT pinned, like build()
    bt_src = rig.res_bt[:B].clone()

    def t_old():
        for _ in range(steps):
            for _ in range(layers):
                md = rig.launch_metadata()
                rig.launch_fa(md)

    def t_new():
        for _ in range(steps):
            md = rig.launch_metadata()
            for _ in range(layers):
                rig.metadata_buf.copy_(md)
                rig.launch_fa(rig.metadata_buf)

    md_warm = rig.launch_metadata()
    torch.npu.synchronize()

    def t_fa():
        for _ in range(steps):
            for _ in range(layers):
                rig.launch_fa(md_warm)

    def t_pin():
        for _ in range(steps):
            rig.res_cu[: B + 1].copy_(pinned_cu, non_blocking=True)
            rig.res_seqused[:B].copy_(plain_seq, non_blocking=True)
            rig.res_bt[:B].copy_(bt_src, non_blocking=True)
            md = rig.launch_metadata()
            for _ in range(layers):
                rig.metadata_buf.copy_(md)
                rig.launch_fa(rig.metadata_buf)

    results = {}
    for name, fn in [("T-fa", t_fa), ("T-old", t_old), ("T-new", t_new), ("T-pin", t_pin)]:
        per_step = timed(fn) / steps
        results[name] = per_step
        print(f"  [T] {name:6s} {per_step * 1e3:8.1f} ms/step")
    ok = results["T-new"] < max(2 * results["T-old"], results["T-old"] + 0.05) and results[
        "T-pin"
    ] < max(2 * results["T-old"], results["T-old"] + 0.05)
    print(f"  [T] {'GREEN' if ok else 'RED'} (gate: T-new and T-pin within 2x of T-old)")
    if not ok:
        print("  [T] the slow component is what the slowest mix adds over the next one down")
    return ok


def stage_m(rig: Rig) -> bool:
    """The NUM_SPEC=3 first-step recipe: prefill TND metadata calls, then
    draft-shaped PA_BBND metadata calls, all from resident slices, repeated.

    Deliberately metadata-centric: the crash named QuantFlashAttnMetadata, so
    this reproduces the metadata task sequence of that step. The TND FA and
    the write-path ops that also sat on the stream are NOT here - if this
    stays GREEN on a machine where the smoke still crashes, add them next."""
    print("== stage M: TND prefill <-> PA_BBND decode metadata interleave x200 ==")
    dev = torch.device("npu")
    total = sum(PREFILL_LENS)
    cu_prefill = torch.zeros(len(PREFILL_LENS) + 1, dtype=torch.int32, device=dev)
    cu_prefill[1:] = torch.cumsum(torch.tensor(PREFILL_LENS, dtype=torch.int32), 0).npu()
    # TND v_descale: sum(ceil(S/64)) token rows. The AICPU check divides dim0
    # by (NKV * D * 2), i.e. it sees the tensor flattened - hand it flat so the
    # check passes whether or not the adapter flattens.
    t_rows = sum((s + 63) // 64 for s in PREFILL_LENS)
    v_descale_tnd = torch.ones(t_rows * NKV * D * 2, dtype=torch.uint8, device=dev).view(E8M0)
    tnd_args = {
        "cu_seqlens_q": cu_prefill,
        "cu_seqlens_kv": cu_prefill,
        "mask_mode": 3,
        "max_seqlen_q": max(PREFILL_LENS),
        "max_seqlen_kv": max(PREFILL_LENS),
        "layout_q": "TND",
        "layout_q_descale": "TND",
        "layout_kv": "TND",
        "layout_out": "TND",
    }
    # Draft decode shape: q_len=1 rows but max_seqlen_q=4 (decode_threshold
    # with NUM_SPEC=3), seqused right after that prefill.
    draft_seqused = rig.res_seqused.clone()
    draft_seqused[:B] = torch.tensor([x + 1 for x in PREFILL_LENS], dtype=torch.int32)
    draft_args = rig.decode_args()
    draft_args["seqused_kv"] = draft_seqused[:B]
    draft_args["max_seqlen_q"] = 4
    draft_args["max_seqlen_kv"] = max(PREFILL_LENS) + 1

    try:
        for i in range(200):
            for _ in range(16):  # the 16 prefill layers
                torch.ops._C_ascend.npu_quant_flash_attn_metadata(
                    NQ, NKV, D, v_descale=v_descale_tnd, **tnd_args
                )
            for step in range(2):  # draft steps 1 and 2
                md = torch.ops._C_ascend.npu_quant_flash_attn_metadata(
                    NQ, NKV, D, v_descale=rig.v_scale.view(E8M0), **draft_args
                )
                rig.metadata_buf.copy_(md)
                # The metadata op and the main op must take the SAME args -
                # a mismatch is undefined behavior, so no launch_fa() here.
                torch.ops._C_ascend.npu_quant_flash_attn(
                    rig.q_fp8.view(FP8),
                    rig.k_fp8.view(FP8),
                    rig.v_fp8.view(FP8),
                    rig.q_descale.contiguous().view(E8M0),
                    rig.k_scale.view(E8M0),
                    rig.v_scale.view(E8M0),
                    rig.metadata_buf,
                    SCALE,
                    block_table=rig.res_bt[:B],
                    attn_mask=rig.mask,
                    **draft_args,
                )
            if (i + 1) % 50 == 0:
                torch.npu.synchronize()
                print(f"  [M] {i + 1}/200 alive")
        torch.npu.synchronize()
    except Exception:
        traceback.print_exc()
        print("  [M] RED - the NUM_SPEC=3 crash reproduces with this recipe")
        return False
    print("  [M] GREEN")
    return True


def stage_b(rig: Rig) -> bool:
    print("== stage B: batch rows vs solo runs (regression-3 shape) ==")
    md = rig.launch_metadata()
    batch_out = rig.launch_fa(md)
    torch.npu.synchronize()
    ok = True
    for b in range(B):
        solo_cu = torch.arange(2, dtype=torch.int32, device="npu")
        solo_args = rig.decode_args(rows=1)
        solo_args["cu_seqlens_q"] = solo_cu
        solo_args["seqused_kv"] = rig.res_seqused[b : b + 1]
        solo_args["max_seqlen_kv"] = SEQUSED[b]
        solo_md = torch.ops._C_ascend.npu_quant_flash_attn_metadata(
            NQ, NKV, D, v_descale=rig.v_scale.view(E8M0), **solo_args
        )
        solo_out = torch.ops._C_ascend.npu_quant_flash_attn(
            rig.q_fp8[b : b + 1].view(FP8),
            rig.k_fp8.view(FP8),
            rig.v_fp8.view(FP8),
            rig.q_descale[:, b : b + 1].contiguous().view(E8M0),
            rig.k_scale.view(E8M0),
            rig.v_scale.view(E8M0),
            solo_md,
            SCALE,
            block_table=rig.res_bt[b : b + 1],
            attn_mask=rig.mask,
            **solo_args,
        )
        torch.npu.synchronize()
        # Different batch shapes tile differently, so reductions may differ in
        # ULPs - bitwise equality is too strict here. A real row-mixing bug
        # (regression 3 moved a chosen-token logprob by 1.9) is orders of
        # magnitude above this gate.
        diff = (batch_out[b].float() - solo_out[0].float()).abs().max().item()
        finite = bool(torch.isfinite(batch_out[b].float()).all())
        close = diff < 2e-2
        print(f"  [B] row {b}: max|batch-solo|={diff:.3e} close={close} finite={finite}")
        ok = ok and close and finite
    print(f"  [B] {'GREEN' if ok else 'RED'} (gate: every row within 2e-2 of its solo run)")
    return ok


def main() -> int:
    global FP8, E8M0
    import os

    import torch_npu  # noqa: F401

    torch.npu.set_device(int(os.environ.get("QFA_TEST_DEVICE", "0")))
    # The aclnn entry points live in the custom opp package, not the system
    # libopapi.so - point the loader at it BEFORE importing vllm_ascend_C,
    # exactly like test_qfa_graph_interleave.py does.
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
    import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: F401

    FP8 = torch.float8_e4m3fn
    E8M0 = torch.float8_e8m0fnu

    print("[INFO] building engine-shaped rig (no weights)")
    rig = Rig()
    results = {}
    for name, fn in [("L", stage_l), ("T", stage_t), ("B", stage_b), ("M", stage_m)]:
        try:
            results[name] = fn(rig)
        except Exception:
            traceback.print_exc()
            results[name] = False
        print()
    for name in ("L", "T", "B", "M"):
        print(f"[{'GREEN' if results[name] else 'RED'}] stage {name}")
    print(
        "[VERDICT] L RED -> AICPU op slow in this shape | T RED -> task-mix overhead | "
        "B RED -> paged batch bug | M RED -> crash recipe reproduced | "
        "all GREEN -> mechanisms fine in isolation, profile the smoke next"
    )
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
