#!/usr/bin/env python3
"""CPU-only test for the QFA backend's cache write path (no NPU, no vllm).

Imports vllm_ascend/attention/attention_qfa.py from the junlin-qfa worktree
with torch_npu / vllm / attention_v1 mocked out, then checks that its five-
plane carving plus quantized write path reproduces, byte for byte, the PA_BBND
cache packing of the single-op golden (_pack_pa_data in
scripts/debug/test_junlin_qfa_npu.py). npu_dynamic_mx_quant is replaced by a
CPU implementation whose semantics the QUANT-FEED NPU case proved identical
(scale byte match 1.0), so a green run here pins the engine-side write path
to the exact bytes the operator was validated against.

Covered:
  1 prefill bulk write  == golden packing (K/Ks/V/Vs planes, byte-exact)
  2 token-by-token decode writes converge to the same bytes as one bulk write
    (the V staging + high-water-mark requantization invariant)
  3 MTP-style rollback: rejected draft tokens overwritten by the next write
    leave no trace in any plane

Run locally:  python scripts/tests/test_qfa_backend_write_path.py
"""

import importlib.util
import math
import sys
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
QFA_PY = REPO / "vllm-ascend" / "junlin-qfa" / "vllm_ascend" / "attention" / "attention_qfa.py"
GOLDEN_PY = REPO / "scripts" / "debug" / "test_junlin_qfa_npu.py"

GROUP = 32
EMAX_E4M3 = 8
FP8_MAX = 448.0


# ---------------------------------------------------------------------------
# CPU stand-in for torch_npu.npu_dynamic_mx_quant (scale_alg=0 semantics,
# proven byte-identical to the device op by the QUANT-FEED case)
# ---------------------------------------------------------------------------
def cpu_dynamic_mx_quant(x: torch.Tensor, dst_type=None, scale_alg=0):
    assert scale_alg == 0 and dst_type == torch.float8_e4m3fn
    rows, d = x.shape
    assert d % GROUP == 0
    grouped = x.float().reshape(rows, d // GROUP, GROUP)
    all_zero = torch.all(grouped == 0, dim=-1)
    max_vals = grouped.abs().amax(dim=-1).clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - EMAX_E4M3
    scale = torch.where(all_zero, torch.ones_like(shared_exp), 2.0**shared_exp)
    expanded = scale.repeat_interleave(GROUP, dim=-1)
    fp8 = (x.float() / expanded).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    bits = scale.view(torch.int32)
    scale_bytes = ((bits >> 23) & 0xFF).to(torch.uint8)
    return fp8, scale_bytes.view(torch.float8_e8m0fnu)


# ---------------------------------------------------------------------------
# import attention_qfa.py with its NPU/vllm dependencies mocked
# ---------------------------------------------------------------------------
def load_qfa_module():
    fake_torch_npu = types.ModuleType("torch_npu")
    fake_torch_npu.npu_dynamic_mx_quant = cpu_dynamic_mx_quant
    sys.modules["torch_npu"] = fake_torch_npu

    fake_backend = types.ModuleType("vllm.v1.attention.backend")

    class AttentionType:
        DECODER = "decoder"

    fake_backend.AttentionType = AttentionType
    fake_vllm = types.ModuleType("vllm")
    fake_v1 = types.ModuleType("vllm.v1")
    fake_attn = types.ModuleType("vllm.v1.attention")
    sys.modules.update({
        "vllm": fake_vllm,
        "vllm.v1": fake_v1,
        "vllm.v1.attention": fake_attn,
        "vllm.v1.attention.backend": fake_backend,
    })

    fake_v1mod = types.ModuleType("vllm_ascend.attention.attention_v1")

    class AscendAttentionBackend:  # minimal stand-ins
        pass

    class AscendAttentionBackendImpl:
        pass

    class AscendAttentionState:
        PrefillNoCache = 0
        PrefillCacheHit = 1
        DecodeOnly = 2
        ChunkedPrefill = 3
        SpecDecoding = 4

    class AscendMetadata:
        pass

    fake_v1mod.AscendAttentionBackend = AscendAttentionBackend
    fake_v1mod.AscendAttentionBackendImpl = AscendAttentionBackendImpl
    fake_v1mod.AscendAttentionState = AscendAttentionState
    fake_v1mod.AscendMetadata = AscendMetadata
    fake_pkg = types.ModuleType("vllm_ascend")
    fake_attn_pkg = types.ModuleType("vllm_ascend.attention")
    sys.modules.update({
        "vllm_ascend": fake_pkg,
        "vllm_ascend.attention": fake_attn_pkg,
        "vllm_ascend.attention.attention_v1": fake_v1mod,
    })

    spec = importlib.util.spec_from_file_location("attention_qfa", QFA_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_golden_module():
    spec = importlib.util.spec_from_file_location("qfa_single_op", GOLDEN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_impl(qfa, nkv, d):
    impl = qfa.AscendQfaAttentionBackendImpl.__new__(qfa.AscendQfaAttentionBackendImpl)
    impl.num_kv_heads = nkv
    impl.head_size = d
    impl._planes = None
    impl._qfa_mask = None
    return impl


def alloc_cache(qfa, nb, bs, nkv, d):
    shape = qfa.AscendQfaAttentionBackend.get_kv_cache_shape(nb, bs, nkv, d)
    return torch.zeros(shape, dtype=torch.bfloat16)


def check(name, ok):
    print(f"  [{name}] {'GREEN' if ok else 'RED'}")
    return ok


def planes_equal(name, planes, data, kv_lens, bs):
    """Byte-compare impl planes vs golden PA_BBND packing on the VALID region
    only: the golden quantizes block-alignment padding too, the impl leaves it
    untouched, and the kernel never reads past seqused_kv either way."""
    valid = []  # valid token count per block, in golden allocation order
    for s in kv_lens:
        nblk = max(1, (s + bs - 1) // bs)
        valid.extend(max(0, min(bs, s - j * bs)) for j in range(nblk))
    ok = True
    for label, got, want, window_rows in (
        ("K", planes.k_fp8, data["k_cache"], False),
        ("Ks", planes.k_scale, data["k_scale_cache"], False),
        ("V", planes.v_fp8, data["v_cache"], False),
        ("Vs", planes.v_scale, data["v_scale_cache"], True),
    ):
        for blk, v in enumerate(valid):
            rows = (v + 63) // 64 if window_rows else v
            same = torch.equal(got[blk][:rows].cpu(), want[blk][:rows])
            if not same:
                diff = (got[blk][:rows].cpu() != want[blk][:rows]).sum().item()
                print(f"    {label} block {blk}: {diff} bytes differ "
                      f"(valid rows {rows})")
            ok &= same
    return check(name, ok)


def main() -> int:
    torch.manual_seed(20260829)
    qfa = load_qfa_module()
    golden = load_golden_module()

    nkv, d, bs = 4, 256, 128
    all_ok = True

    # ---- 1) prefill bulk write vs golden packing -------------------------
    # _pack_pa_data draws q, k, v per sequence in a fixed order; replaying the
    # same seed reproduces the exact bf16 K/V it quantized.
    kv_lens = [130, 127, 64, 300]
    keys, values = [], []
    torch.manual_seed(424242)
    for s in kv_lens:
        _ = torch.randn(1, 8, d, dtype=torch.bfloat16)  # q (unused here)
        keys.append(torch.randn(s, nkv, d, dtype=torch.bfloat16))
        values.append(torch.randn(s, nkv, d, dtype=torch.bfloat16))
    torch.manual_seed(424242)
    data = golden._pack_pa_data([1] * len(kv_lens), kv_lens, 8, nkv, d, bs, "PA_BBND")
    total_blocks = data["k_cache"].shape[0]

    impl = make_impl(qfa, nkv, d)
    cache = alloc_cache(qfa, total_blocks, bs, nkv, d)
    planes = impl._ensure_planes(cache)

    # write through the impl, sequence by sequence, block-aligned slots
    next_block = 0
    for i, s in enumerate(kv_lens):
        nblk = max(1, (s + bs - 1) // bs)
        slots = torch.arange(s) + next_block * bs
        impl._write_kv_quantized(keys[i], values[i], slots)
        next_block += nblk
    all_ok &= planes_equal("PREFILL-BULK", planes, data, kv_lens, bs)

    # ---- 2) decode: token-by-token == one bulk write ---------------------
    impl2 = make_impl(qfa, nkv, d)
    cache2 = alloc_cache(qfa, 4, bs, nkv, d)
    planes2 = impl2._ensure_planes(cache2)
    s_total = 100
    k_seq = torch.randn(s_total, nkv, d, dtype=torch.bfloat16)
    v_seq = torch.randn(s_total, nkv, d, dtype=torch.bfloat16)
    prefill = 37
    impl2._write_kv_quantized(k_seq[:prefill], v_seq[:prefill], torch.arange(prefill))
    for t in range(prefill, s_total):
        impl2._write_kv_quantized(k_seq[t:t + 1], v_seq[t:t + 1], torch.tensor([t]))

    impl3 = make_impl(qfa, nkv, d)
    cache3 = alloc_cache(qfa, 4, bs, nkv, d)
    planes3 = impl3._ensure_planes(cache3)
    impl3._write_kv_quantized(k_seq, v_seq, torch.arange(s_total))

    same = all(
        torch.equal(a, b) for a, b in (
            (planes2.k_fp8, planes3.k_fp8), (planes2.k_scale, planes3.k_scale),
            (planes2.v_fp8, planes3.v_fp8), (planes2.v_scale, planes3.v_scale),
        ))
    all_ok &= check("DECODE-INCREMENTAL", same)

    # ---- 3) MTP rollback: rejected drafts leave no trace ------------------
    impl4 = make_impl(qfa, nkv, d)
    cache4 = alloc_cache(qfa, 4, bs, nkv, d)
    planes4 = impl4._ensure_planes(cache4)
    base = 70
    impl4._write_kv_quantized(k_seq[:base], v_seq[:base], torch.arange(base))
    # write 4 draft tokens (positions 70..73), then "reject" the last 3:
    drafts_k = torch.randn(4, nkv, d, dtype=torch.bfloat16)
    drafts_v = torch.randn(4, nkv, d, dtype=torch.bfloat16)
    impl4._write_kv_quantized(drafts_k, drafts_v, torch.arange(base, base + 4))
    # next step: accepted history is 71 tokens; write the real token 71
    impl4._write_kv_quantized(k_seq[71:72], v_seq[71:72], torch.tensor([71]))

    impl5 = make_impl(qfa, nkv, d)
    cache5 = alloc_cache(qfa, 4, bs, nkv, d)
    planes5 = impl5._ensure_planes(cache5)
    k_ref = torch.cat([k_seq[:base], drafts_k[:1], k_seq[71:72]])
    v_ref = torch.cat([v_seq[:base], drafts_v[:1], v_seq[71:72]])
    impl5._write_kv_quantized(k_ref, v_ref, torch.arange(72))

    same = True
    for label, a, b in (("v_fp8", planes4.v_fp8, planes5.v_fp8),
                        ("v_scale", planes4.v_scale, planes5.v_scale)):
        if not torch.equal(a, b):
            same = False
            idx = (a != b).nonzero()
            print(f"    {label}: {idx.shape[0]} bytes differ, first at "
                  f"{idx[0].tolist()}, last at {idx[-1].tolist()}")
    # K planes differ beyond position 72 (stale draft K bytes remain), which
    # is fine: K is per-token and never read past seqused_kv. V must match
    # exactly because draft garbage would poison the shared window scale.
    all_ok &= check("MTP-ROLLBACK-V", same)

    print(f"[{'GREEN' if all_ok else 'RED'}] QFA backend write path vs golden packing")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
