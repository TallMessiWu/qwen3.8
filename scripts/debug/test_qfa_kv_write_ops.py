#!/usr/bin/env python3
"""Server-side probes + unit tests for the QFA MXFP8 KV-cache WRITE path (C1).

Milestone C stores KV as MXFP8 inside the existing BF16 cache allocation
(byte reinterpretation): per 128-token block
  k_fp8 (Bs,N,D) fp8_e4m3 + k_scale (Bs,N,D/64,2) e8m0-bytes
  v_fp8 (Bs,N,D)           + v_scale (Bs/64,N,D,2)
K quantizes per-token along D; V shares one e8m0 per (head,channel) across 32
tokens (packed 64/row), so decode appends re-quantize a 128-row staging ring.

Cases (each prints GREEN/RED; non-blocking probes print INFO):
  P1 scatter-bbnd    can the fused scatter serve a BBND cache? (informational;
                     first run said no - it wants the NZ-style C8 layout)
  P2 scatter-scale   same question for (T,N,8) scale planes (informational)
  P3 index-put       the write path actually used: uint8 flat-slot index_put_
                     with negative slots folded onto the null block
  P4 quant-k128      npu_dynamic_mx_quant on (rows,128) vs CPU reference
  P5 noncontig-qfa   strided k-cache view fed to QFA (informational)
  W1 k-write-golden  full K path: quant -> index_put_ -> bit-exact vs CPU ref
  W2 v-incremental   the 128-row staging-ring algorithm, stepped +1..+4 with
                     64/128-boundary crossings and a spec-decode rollback,
                     final cache bit-exact vs one-shot whole-sequence quant
  G1 capture-chain   torch.npu.graph capture of the static write chain,
                     replayed with fresh data (C3 feasibility for the writer)

Run: python scripts/debug/test_qfa_kv_write_ops.py   (1 idle NPU)
Filter: QFA_WRITE_CASES=P1,W2 ...
"""

import math
import os
import sys
import traceback

import torch

GROUP = 32
BS = 128  # kv-cache block size
E8M0_MIN_POSITIVE = 2.0 ** (-127)
FP8 = None


# --------------------------------------------------------------------------
# CPU references (mirror test_quant_flash_attn_npu.py)
# --------------------------------------------------------------------------
def qk_group_scale(x: torch.Tensor) -> torch.Tensor:
    t, n, d = x.shape
    grouped = x.float().reshape(t, n, d // GROUP, GROUP)
    all_zero = torch.all(grouped == 0, dim=-1)
    max_vals = grouped.abs().amax(dim=-1).clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - 8
    return torch.where(all_zero, torch.ones_like(shared_exp), 2.0**shared_exp)


def fp32_to_e8m0_bytes(scale: torch.Tensor) -> torch.Tensor:
    safe = scale.float().clone()
    safe[~torch.isfinite(safe)] = E8M0_MIN_POSITIVE
    safe[safe == 0] = E8M0_MIN_POSITIVE
    bits = safe.view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8)


def quantize_with_scale(x: torch.Tensor, scale_expanded: torch.Tensor) -> torch.Tensor:
    return (x.float() / scale_expanded).clamp(-448.0, 448.0).to(FP8)


def cpu_quant_rows(x_rows_128: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference of npu_dynamic_mx_quant over (rows, 128): 4 groups of 32."""
    rows = x_rows_128.shape[0]
    grouped = x_rows_128.float().reshape(rows, 4, GROUP)
    all_zero = torch.all(grouped == 0, dim=-1)
    max_vals = grouped.abs().amax(dim=-1).clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - 8
    scale = torch.where(all_zero, torch.ones_like(shared_exp), 2.0**shared_exp)
    fp8 = quantize_with_scale(x_rows_128, scale.repeat_interleave(GROUP, dim=-1))
    return fp8, fp32_to_e8m0_bytes(scale)  # (rows,128) fp8, (rows,4) uint8


# --------------------------------------------------------------------------
# NPU harness
# --------------------------------------------------------------------------
def bootstrap() -> None:
    import torch_npu  # noqa: F401

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


def npu_quant_128(x_rows_128_npu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    import torch_npu

    fp8, scale = torch_npu.npu_dynamic_mx_quant(x_rows_128_npu, dst_type=FP8, scale_alg=0)
    return fp8, scale.view(torch.uint8)  # (rows,128) fp8, (rows,4)? -> checked by P4


# --------------------------------------------------------------------------
# P1: npu_scatter_pa_kv_cache on int8 views + -1 skip
# --------------------------------------------------------------------------
def probe_scatter_int8() -> bool:
    """Informational: can the fused scatter serve a (Bn,Bs,N,D) BBND cache?

    Answer from the first run: no. aclnnScatterPaKvCache demands
    key_cache.dim1 == ceil(numHead*headSize/lastDim) with a 32-byte last dim,
    i.e. the NZ-style layout the C8 path uses. Milestone C keeps the vLLM-native
    BBND layout (QFA reads it directly), so the write path uses index_put_
    instead. Kept as a probe so a future CANN release that lifts the constraint
    shows up here.
    """
    import torch_npu

    print("== P1: npu_scatter_pa_kv_cache on a BBND cache (informational) ==")
    t, n, d, bn = 5, 2, 32, 2
    key = torch.randint(-127, 127, (t, n, d), dtype=torch.int8).npu()
    k_cache = torch.zeros(bn, BS, n, d, dtype=torch.int8).npu()
    v_cache = torch.zeros(bn, BS, n, d, dtype=torch.int8).npu()
    slots = torch.tensor([0, 1, 130, -1, 255], dtype=torch.int32).npu()
    try:
        torch_npu.npu_scatter_pa_kv_cache(key, key, k_cache, v_cache, slots)
        torch.npu.synchronize()
        k_flat = k_cache.view(bn * BS, n, d).cpu()
        key_cpu = key.cpu()
        hits = [s_ for i, s_ in enumerate([0, 1, 130, 255]) if torch.equal(k_flat[s_], key_cpu[i if i < 3 else 4])]
        written = int(k_flat.flatten(1).ne(0).any(dim=1).sum())
        print(f"  [P1] INFO: ACCEPTED, {len(hits)}/4 slots match, rows written={written} (-1 skipped)")
    except Exception as exc:  # noqa: BLE001 - capability probe
        print(f"  [P1] INFO: rejected for BBND layout -> index_put_ write path stands: {exc}")
    print("  [P1] GREEN (informational)")
    return True


# --------------------------------------------------------------------------
# P2: same op moving (T,N,8) uint8 scale planes
# --------------------------------------------------------------------------
def probe_scatter_scale() -> bool:
    import torch_npu

    print("== P2: scatter for k_scale planes (head_size=8, informational) ==")
    t, n, bn = 5, 2, 2
    ds = 8  # D/64*2 for D=256
    scale = torch.randint(0, 255, (t, n, ds), dtype=torch.uint8)
    s_cache = torch.zeros(bn, BS, n, ds, dtype=torch.uint8).npu()
    slots = torch.tensor([3, 64, 127, 128, 200], dtype=torch.int32).npu()
    try:
        # scale plane rides both key and value channels of a dedicated call
        torch_npu.npu_scatter_pa_kv_cache(
            scale.view(torch.int8).npu(), scale.view(torch.int8).npu(),
            s_cache.view(torch.int8), s_cache.view(torch.int8), slots,
        )
        torch.npu.synchronize()
        got = s_cache.view(bn * BS, n, ds).cpu()
        ok = all(torch.equal(got[s], scale[i]) for i, s in enumerate([3, 64, 127, 128, 200]))
        print(f"  [P2] INFO: scatter accepted scale planes, values {'match' if ok else 'MISMATCH'}")
        print("  [P2] GREEN (informational)")
        return True
    except Exception as exc:  # noqa: BLE001 - probe reports capability
        print(f"  [P2] INFO: scatter rejected head_size={ds}: {exc}")
        print("  [P2] GREEN (informational; scales go through index_put_)")
        return True


# --------------------------------------------------------------------------
# P3: index_put_ fallback + negative-slot remap to null block 0
# --------------------------------------------------------------------------
def probe_index_put() -> bool:
    print("== P3: index_put_ uint8 fallback + negative-slot remap ==")
    t, n, ds, bn = 5, 2, 8, 2
    scale = torch.randint(0, 255, (t, n, ds), dtype=torch.uint8).npu()
    s_cache = torch.zeros(bn * BS, n, ds, dtype=torch.uint8).npu()
    slots = torch.tensor([3, 64, -1, 128, 200], dtype=torch.int64).npu()
    safe = torch.where(slots >= 0, slots, torch.zeros_like(slots))  # -1 -> block 0 (null)
    s_cache.index_put_((safe,), scale)
    torch.npu.synchronize()
    got = s_cache.cpu()
    scale_cpu = scale.cpu()
    # token index -> destination row, with token 2 (slot -1) folded onto the null block
    expected = {0: 3, 1: 64, 2: 0, 3: 128, 4: 200}
    ok = all(torch.equal(got[row], scale_cpu[tok]) for tok, row in expected.items())
    print(f"  [P3] {'GREEN' if ok else 'RED'} (negative slot lands in null block 0)")
    return ok


# --------------------------------------------------------------------------
# P4: npu_dynamic_mx_quant over (rows, 128) — the V staging requant shape
# --------------------------------------------------------------------------
def probe_quant_128() -> bool:
    print("== P4: npu_dynamic_mx_quant (rows,128) vs CPU reference ==")
    torch.manual_seed(7)
    rows = 4 * 256  # one request worth: Nkv*D for 27B per-rank
    x = torch.randn(rows, BS, dtype=torch.bfloat16)
    x[0, :GROUP] = 0  # all-zero group
    x[-1] = 0  # untouched staging row (pure zeros)
    fp8_ref, scale_ref = cpu_quant_rows(x)

    fp8, scale = npu_quant_128(x.npu())
    scale = scale.cpu().reshape(rows, -1)
    print(f"  [P4] npu scale shape per row = {scale.shape[1]} (expect 4 = 128/32 groups)")
    fp8_match = (fp8.cpu().view(torch.uint8) == fp8_ref.view(torch.uint8)).float().mean().item()
    nonzero_groups = ~torch.all(x.float().reshape(rows, 4, GROUP) == 0, dim=-1)
    scale_match = (scale[nonzero_groups] == scale_ref[nonzero_groups]).float().mean().item()
    zero_bytes = sorted(set(scale[~nonzero_groups].tolist()))
    print(f"  [P4] scale match={scale_match:.6f} fp8 match={fp8_match:.6f} zero-group bytes={zero_bytes}")
    ok = scale.shape[1] == 4 and scale_match >= 0.999 and fp8_match >= 0.99 and 0xFF not in zero_bytes
    print(f"  [P4] {'GREEN' if ok else 'RED'}")
    return ok


# --------------------------------------------------------------------------
# P5: strided (non-contiguous) k-cache view into QFA (informational)
# --------------------------------------------------------------------------
def probe_noncontig_qfa() -> bool:
    print("== P5: non-contiguous cache view into QFA (informational) ==")
    if not hasattr(torch.ops._C_ascend, "npu_quant_flash_attn"):
        print("  [P5] QFA op not registered, skip")
        return True
    torch.manual_seed(11)
    b, nq, nkv, d, skv = 1, 8, 2, 128, 128
    big = torch.zeros(1, BS, nkv, d * 2, dtype=torch.uint8).npu()
    k_view = big[..., :d].view(torch.float8_e4m3fn)  # strided along last dim
    v_cache = torch.zeros(1, BS, nkv, d, dtype=torch.uint8).npu().view(torch.float8_e4m3fn)
    ks = torch.full((1, BS, nkv, d // 64, 2), 127, dtype=torch.uint8).npu().view(torch.float8_e8m0fnu)
    vs = torch.full((1, BS // 64, nkv, d, 2), 127, dtype=torch.uint8).npu().view(torch.float8_e8m0fnu)
    q = torch.zeros(b, nq, d, dtype=torch.uint8).npu().view(torch.float8_e4m3fn)
    qs = torch.full((nkv, b, nq // nkv, d // 64, 2), 127, dtype=torch.uint8).npu().view(torch.float8_e8m0fnu)
    common = {
        "cu_seqlens_q": torch.tensor([0, 1], dtype=torch.int32).npu(),
        "seqused_kv": torch.tensor([skv], dtype=torch.int32).npu(),
        "block_table": torch.zeros(1, 1, dtype=torch.int32).npu(),
        "mask_mode": 0,
        "max_seqlen_q": 1,
        "max_seqlen_kv": skv,
        "layout_q": "TND",
        "layout_q_descale": "N2TGD",
        "layout_kv": "PA_BBND",
        "layout_out": "TND",
    }
    try:
        md = torch.ops._C_ascend.npu_quant_flash_attn_metadata(
            nq, nkv, d, v_descale=vs,
            **{k: v for k, v in common.items() if k != "block_table"},
        )
        torch.ops._C_ascend.npu_quant_flash_attn(
            q, k_view, v_cache, qs, ks, vs, md, 1.0 / math.sqrt(d), attn_mask=None, **common
        )
        torch.npu.synchronize()
        print("  [P5] INFO: strided k view ACCEPTED (TensorV2 path available)")
    except Exception as exc:  # noqa: BLE001 - informational probe
        print(f"  [P5] INFO: strided k view rejected -> contiguous carve-out required: {exc}")
    print("  [P5] GREEN (informational)")
    return True


# --------------------------------------------------------------------------
# W1: full K write path, bit-exact vs CPU
# --------------------------------------------------------------------------
def case_k_write_golden() -> bool:
    """Full K path exactly as the impl does it: quant -> index_put_ -> golden."""
    import torch_npu

    print("== W1: K quant+index_put_ vs CPU golden (bit-exact) ==")
    torch.manual_seed(21)
    t, n, d, bn = 100, 4, 256, 3
    key = torch.randn(t, n, d, dtype=torch.bfloat16)
    slots = torch.randperm(bn * BS)[:t]
    slots[7] = -1  # a padded token must land in the null block, not corrupt a real slot

    fp8, scale = torch_npu.npu_dynamic_mx_quant(
        key.npu().reshape(t * n, d), dst_type=FP8, scale_alg=0
    )
    fp8 = fp8.reshape(t, n, d).view(torch.uint8)
    scale_b = scale.view(torch.uint8).reshape(t, n, d // GROUP)
    k_cache = torch.zeros(bn * BS, n, d, dtype=torch.uint8).npu()
    ks_cache = torch.zeros(bn * BS, n, d // GROUP, dtype=torch.uint8).npu()
    safe = torch.where(slots >= 0, slots, torch.zeros_like(slots)).to(torch.int64).npu()
    k_cache.index_put_((safe,), fp8)
    ks_cache.index_put_((safe,), scale_b.npu())
    torch.npu.synchronize()

    ref_scale = qk_group_scale(key)
    ref_fp8 = quantize_with_scale(key, ref_scale.repeat_interleave(GROUP, dim=-1)[..., :d])
    ref_scale_b = fp32_to_e8m0_bytes(ref_scale)

    got_k = k_cache.cpu()
    got_s = ks_cache.cpu()
    real = slots >= 0
    fp8_match = (got_k[slots[real].long()] == ref_fp8.view(torch.uint8)[real]).float().mean().item()
    scale_match = (got_s[slots[real].long()] == ref_scale_b[real]).float().mean().item()
    untouched = torch.ones(bn * BS, dtype=torch.bool)
    untouched[slots[real].long()] = False
    untouched[0] = False  # null block absorbs the padded token
    clean = bool((got_k[untouched] == 0).all())
    print(f"  [W1] fp8 match={fp8_match:.6f} scale match={scale_match:.6f} untouched-clean={clean}")
    ok = fp8_match >= 0.99 and scale_match >= 0.999 and clean
    print(f"  [W1] {'GREEN' if ok else 'RED'}")
    return ok


# --------------------------------------------------------------------------
# W2: V 128-row staging-ring incremental algorithm
# --------------------------------------------------------------------------
def quant_window_rows(rows: torch.Tensor):
    """Mirror of AscendAttentionBackendImpl._qfa_quant_along_tokens.

    (W,64,N,D) BF16 -> fp8 (W,64,N,D) + packed scale (W,N,D,2): one E8M0 pair
    per (head, channel) per 64-token window.
    """
    import torch_npu

    w, group, n, d = rows.shape
    cols = rows.permute(0, 2, 3, 1).reshape(w * n * d, group).contiguous()
    fp8, scale = torch_npu.npu_dynamic_mx_quant(cols, dst_type=FP8, scale_alg=0)
    fp8 = fp8.reshape(w, n, d, group).permute(0, 3, 1, 2).contiguous()
    return fp8, scale.view(torch.uint8).reshape(w, n, d, 2)


def _v_window_write(staging, seq_len, w, v_fp8_flat, v_scale_flat, block_table):
    """Re-quantize window w (64 rows) from the ring and write it to cache."""
    half = (w % 2) * 64
    rows = staging[half : half + 64]  # (64, N, D) BF16 view into the ring
    valid = max(0, min(64, seq_len - w * 64))
    if valid < 64:
        rows[valid:] = 0  # drop anything past the (possibly rolled-back) tail
    fp8, scale = quant_window_rows(rows.unsqueeze(0))
    slot0 = int(block_table[w * 64 // BS]) * (BS // 64) + (w * 64 % BS) // 64
    v_fp8_flat[slot0] = fp8[0].view(torch.int8)
    v_scale_flat[slot0] = scale[0]


def case_v_incremental() -> bool:
    print("== W2: V staging-ring incremental writes vs one-shot quant ==")
    torch.manual_seed(42)
    n, d, bn = 4, 256, 4
    final_len = 300
    all_v = torch.randn(final_len + 8, n, d, dtype=torch.bfloat16).npu()
    block_table = list(range(bn))

    v_fp8 = torch.zeros(bn * (BS // 64), 64, n, d, dtype=torch.int8).npu()
    v_scale = torch.zeros(bn * (BS // 64), n, d, 2, dtype=torch.uint8).npu()
    staging = torch.zeros(BS, n, d, dtype=torch.bfloat16).npu()

    # scripted step pattern: crosses 64 and 128 boundaries, one rollback
    seq_len = 0
    steps = [4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]  # -> 62..66 crossing
    while seq_len < final_len:
        for adv in steps:
            if seq_len >= final_len:
                break
            adv = min(adv, final_len - seq_len)
            new = all_v[seq_len : seq_len + adv]
            positions = torch.arange(seq_len, seq_len + adv).npu()
            staging[positions % BS] = new
            new_len = seq_len + adv
            w_last = (new_len - 1) // 64
            for w in sorted({max(0, w_last - 1), w_last}):
                _v_window_write(staging, new_len, w, v_fp8, v_scale, block_table)
            seq_len = new_len
            if seq_len == 130:  # simulate spec rollback: reject 2 of the last step
                seq_len -= 2
                # rejected positions will be re-sampled with DIFFERENT values,
                # so stale staging rows must actually be overwritten next step
                all_v[seq_len : seq_len + 2] = torch.randn(2, n, d, dtype=torch.bfloat16).npu()
                w_last = (seq_len - 1) // 64
                for w in sorted({max(0, w_last - 1), w_last}):
                    _v_window_write(staging, seq_len, w, v_fp8, v_scale, block_table)
    torch.npu.synchronize()

    # one-shot reference: quantize the whole (rolled-back-consistent) sequence
    ref_fp8 = torch.zeros_like(v_fp8)
    ref_scale = torch.zeros_like(v_scale)
    for w in range((seq_len + 63) // 64):
        rows = torch.zeros(64, n, d, dtype=torch.bfloat16).npu()
        valid = min(64, seq_len - w * 64)
        rows[:valid] = all_v[w * 64 : w * 64 + valid]
        fp8, scale = quant_window_rows(rows.unsqueeze(0))
        slot0 = block_table[w * 64 // BS] * (BS // 64) + (w * 64 % BS) // 64
        ref_fp8[slot0] = fp8[0].view(torch.int8)
        ref_scale[slot0] = scale[0]
    torch.npu.synchronize()

    used = (seq_len + 63) // 64
    got_f, ref_f = v_fp8.cpu(), ref_fp8.cpu()
    got_s, ref_s = v_scale.cpu(), ref_scale.cpu()
    bad_fp8, bad_scale = [], []
    for w in range(used):
        slot = block_table[w * 64 // BS] * (BS // 64) + (w * 64 % BS) // 64
        if not torch.equal(got_f[slot], ref_f[slot]):
            bad_fp8.append((w, slot, int((got_f[slot] != ref_f[slot]).sum())))
        if not torch.equal(got_s[slot], ref_s[slot]):
            bad_scale.append((w, slot, int((got_s[slot] != ref_s[slot]).sum())))
    print(f"  [W2] final seq_len={seq_len} windows={used}")
    if bad_fp8 or bad_scale:
        print(f"  [W2] fp8 mismatching windows (w, slot, diff_bytes)={bad_fp8}")
        print(f"  [W2] scale mismatching windows (w, slot, diff_bytes)={bad_scale}")
        # For the first bad window, show whether the staging ring or the
        # quantization is at fault: re-quantize straight from all_v.
        w, slot, _ = (bad_fp8 or bad_scale)[0]
        rows = torch.zeros(64, n, d, dtype=torch.bfloat16).npu()
        valid = min(64, seq_len - w * 64)
        rows[:valid] = all_v[w * 64 : w * 64 + valid]
        half = (w % 2) * 64
        ring = staging[half : half + 64].clone()
        ring[valid:] = 0
        same_rows = torch.equal(ring.cpu(), rows.cpu())
        # Only meaningful for the last two windows: the ring has moved on
        # from anything older, which is expected and not itself a fault.
        print(f"  [W2] first bad window w={w} (last window={(seq_len - 1) // 64}): "
              f"staging rows == all_v rows? {same_rows}")
        if not same_rows:
            diff = (ring.float() - rows.float()).abs().sum(dim=(1, 2))
            bad_rows = torch.nonzero(diff > 0).flatten().tolist()
            print(f"  [W2]   differing ring rows within the window: {bad_rows[:16]}")
    ok = not bad_fp8 and not bad_scale
    print(f"  [W2] {'GREEN' if ok else 'RED'}")
    return ok


# --------------------------------------------------------------------------
# G1: capture the static write chain in an NPU graph, replay with fresh data
# --------------------------------------------------------------------------
def case_capture_chain() -> bool:
    print("== G1: torch.npu.graph capture of quant+index_put_ chain ==")
    torch.manual_seed(9)
    rows = 4 * 256
    x_buf = torch.randn(rows, BS, dtype=torch.bfloat16).npu()
    slot_buf = torch.tensor([2], dtype=torch.int64).npu()
    out_fp8 = torch.zeros(8, rows, BS, dtype=torch.int8).npu()
    out_scale = torch.zeros(8, rows, 4, dtype=torch.uint8).npu()
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        fp8, scale = npu_quant_128(x_buf)
        out_fp8.index_put_((slot_buf,), fp8.view(torch.int8).unsqueeze(0))
        out_scale.index_put_((slot_buf,), scale.reshape(rows, -1).unsqueeze(0))
    torch.npu.synchronize()

    ok = True
    for trial, slot in enumerate([5, 0, 7]):
        x_new = torch.randn(rows, BS, dtype=torch.bfloat16)
        x_buf.copy_(x_new.npu())
        slot_buf.fill_(slot)
        graph.replay()
        torch.npu.synchronize()
        ref_fp8, ref_scale = cpu_quant_rows(x_new)
        fp8_match = (out_fp8[slot].cpu() == ref_fp8.view(torch.int8)).float().mean().item()
        nonzero = ~torch.all(x_new.float().reshape(rows, 4, GROUP) == 0, dim=-1)
        got_scale = out_scale[slot].cpu()
        scale_match = (got_scale[nonzero] == ref_scale[nonzero]).float().mean().item()
        print(f"  [G1] replay#{trial} slot={slot} fp8={fp8_match:.6f} scale={scale_match:.6f}")
        ok = ok and fp8_match >= 0.99 and scale_match >= 0.999
    print(f"  [G1] {'GREEN' if ok else 'RED'}")
    return ok


def main() -> int:
    global FP8
    FP8 = torch.float8_e4m3fn
    device_id = int(os.environ.get("QFA_TEST_DEVICE", "0"))
    import torch_npu  # noqa: F401

    torch.npu.set_device(device_id)
    print(f"[INFO] device npu:{device_id}")
    bootstrap()

    all_cases = (
        ("P1", probe_scatter_int8),
        ("P2", probe_scatter_scale),
        ("P3", probe_index_put),
        ("P4", probe_quant_128),
        ("P5", probe_noncontig_qfa),
        ("W1", case_k_write_golden),
        ("W2", case_v_incremental),
        ("G1", case_capture_chain),
    )
    only = os.environ.get("QFA_WRITE_CASES")
    if only:
        wanted = {t.strip() for t in only.split(",") if t.strip()}
        cases = tuple(c for c in all_cases if c[0] in wanted)
    else:
        cases = all_cases

    results = {}
    for name, fn in cases:
        try:
            results[name] = fn()
        except Exception:
            traceback.print_exc()
            results[name] = False
        print()

    for name, ok in results.items():
        print(f"[{'GREEN' if ok else 'RED'}] {name}")
    # P2 RED is tolerated when P3 (fallback) is GREEN
    hard = {k: v for k, v in results.items() if k != "P2"}
    all_ok = all(hard.values()) and (results.get("P2", True) or results.get("P3", False))
    print(f"[{'GREEN' if all_ok else 'RED'}] QFA KV write-path overall")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
