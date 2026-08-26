#!/usr/bin/env python3
"""Server-side smoke test for the vendored quant_flash_attn (QFA) custom op.

Verifies, on a single A5 NPU, that the MXFP8 QuantFlashAttn op integrated into
vllm-ascend (torch.ops._C_ascend.npu_quant_flash_attn{,_metadata}) computes
attention correctly against a CPU golden ported from the ops-transformer test
suite (tests/pytest/qfa_mxfp8_test/common/quant_flash_attn_golden.py):

  case 1 TND      varlen prefill, B=2 S=[128, 200] (non-64-aligned tail group)
  case 2 PA_BBND  paged-attention decode, block_size=128, q_descale N2TGD
  case 3 QUANT    torch_npu.npu_dynamic_mx_quant vs. the CPU reference
                  quantization (validates the online-quant path used by
                  VLLM_ASCEND_ENABLE_QFA_PREFILL, incl. zero-padded groups)
  case 4 27B      Qwen3.8-27B prefill shape repro (B=1 S=1594 Nq=24 Nkv=4
                  D=256): metadata AICPU op and main op run with a synchronize
                  between them, so an AICPU crash points at the exact stage.
                  Bisect knobs: QFA27B_S / QFA27B_NQ / QFA27B_NKV / QFA27B_D
                  (e.g. QFA27B_D=128 checks whether D=256 is the trigger),
                  QFA27B_GOLDEN=0 skips the slow CPU golden (crash-only mode),
                  QFA27B_ITERS=12 additionally fires 12 back-to-back rounds of
                  online-quant+metadata+main with no host sync, mimicking the
                  per-layer launch pattern of a real 27B forward pass.
                  QFA_CASES=27B runs just this case.

Prints [GREEN]/[RED] per case and exits non-zero on any RED. Requires the
freshly built cann-ops-transformer custom package installed under
vllm_ascend/_cann_ops_custom (see scripts in csrc/build_aclnn.sh comments).
"""

import math
import os
import sys
import traceback

import torch

GROUP = 32
FP8 = None  # set after torch import checks
FP8_MAX = 448.0
E8M0_MIN_POSITIVE = 2.0 ** (-127)


# --------------------------------------------------------------------------
# CPU reference quantization (ported from quant_flash_attn_golden.py)
# --------------------------------------------------------------------------
def qk_group_scale(x: torch.Tensor) -> torch.Tensor:
    """Per-token-group scale along D for (T, N, D); returns (T, N, D//32)."""
    t, n, d = x.shape
    assert d % GROUP == 0
    grouped = x.float().reshape(t, n, d // GROUP, GROUP)
    all_zero = torch.all(grouped == 0, dim=-1)
    max_vals = grouped.abs().amax(dim=-1).clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - 8  # emax(e4m3)=8
    return torch.where(all_zero, torch.ones_like(shared_exp), 2.0**shared_exp)


def v_group_scale(x: torch.Tensor) -> torch.Tensor:
    """Per-channel-group scale along S for (S, N, D); returns (S//32, N, D).

    S must already be padded to a multiple of 32 (zeros never raise a max).
    """
    s, n, d = x.shape
    assert s % GROUP == 0
    grouped = x.float().reshape(s // GROUP, GROUP, n, d)
    all_zero = torch.all(grouped == 0, dim=1)
    max_vals = grouped.abs().amax(dim=1).clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - 8
    return torch.where(all_zero, torch.ones_like(shared_exp), 2.0**shared_exp)


def quantize_with_scale(x: torch.Tensor, scale_expanded: torch.Tensor) -> torch.Tensor:
    q = (x.float() / scale_expanded).clamp(-FP8_MAX, FP8_MAX)
    return q.to(FP8)


def fp32_to_e8m0_bytes(scale: torch.Tensor, name: str) -> torch.Tensor:
    """FP32 scale -> uint8 biased exponents; refuses NaN bytes (0xFF)."""
    safe = scale.float().clone()
    safe[~torch.isfinite(safe)] = E8M0_MIN_POSITIVE
    safe[safe == 0] = E8M0_MIN_POSITIVE
    bits = safe.view(torch.int32)
    exp = ((bits >> 23) & 0xFF).to(torch.uint8)
    nan_count = int((exp == 0xFF).sum())
    if nan_count:
        raise ValueError(f"{name}: {nan_count} scales would become e8m0 NaN")
    return exp


def pack_last_pairs(scale: torch.Tensor) -> torch.Tensor:
    """(..., G) -> (..., G//2, 2): adjacent groups pack pairwise."""
    return scale.reshape(*scale.shape[:-1], scale.shape[-1] // 2, 2)


def pack_v_scale_seq(scale_sg_n_d: torch.Tensor) -> torch.Tensor:
    """(Sg, N, D) -> (ceil(Sg/2), N, D, 2): (even, odd) group rows interleave.

    Odd row counts pad with E8M0_MIN_POSITIVE (0.0 would become e8m0 NaN).
    """
    sg, n, d = scale_sg_n_d.shape
    if sg % 2:
        pad = torch.full((1, n, d), E8M0_MIN_POSITIVE, dtype=scale_sg_n_d.dtype)
        scale_sg_n_d = torch.cat([scale_sg_n_d, pad], dim=0)
        sg += 1
    return scale_sg_n_d.reshape(sg // 2, 2, n, d).permute(0, 2, 3, 1).contiguous()


# --------------------------------------------------------------------------
# CPU golden attention (blockwise pipeline port; single sequence, BNSD)
# --------------------------------------------------------------------------
def cpu_golden_one_seq(
    q_fp8: torch.Tensor,  # (Sq, Nq, D) fp8
    k_fp8: torch.Tensor,  # (Skv, Nkv, D) fp8
    v_fp8: torch.Tensor,  # (Skv, Nkv, D) fp8
    q_scale: torch.Tensor,  # (Sq, Nq, D//32) fp32
    k_scale: torch.Tensor,  # (Skv, Nkv, D//32) fp32
    v_scale: torch.Tensor,  # (ceil(Skv/32), Nkv, D) fp32
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    ln2 = math.log(2.0)
    sq, nq, d = q_fp8.shape
    skv, nkv, _ = k_fp8.shape
    group_rep = nq // nkv

    def to_bnsd(x):  # (S, N, ...) -> (1, N, S, ...)
        return x.float().permute(1, 0, *range(2, x.dim())).unsqueeze(0)

    q = to_bnsd(q_fp8)
    k = to_bnsd(k_fp8).repeat_interleave(group_rep, dim=1)
    v = to_bnsd(v_fp8).repeat_interleave(group_rep, dim=1)
    dq = to_bnsd(q_scale).repeat_interleave(GROUP, dim=-1)[..., :d]
    dk = to_bnsd(k_scale).repeat_interleave(group_rep, dim=1).repeat_interleave(GROUP, dim=-1)[..., :d]
    # v_scale (Sg, N, D) -> (1, N, Sg, D) -> expand along S
    dv = v_scale.float().permute(1, 0, 2).unsqueeze(0).repeat_interleave(group_rep, dim=1)
    dv = dv.repeat_interleave(GROUP, dim=2)[:, :, :skv, :]

    q_block, k_block = (128, 128) if d == 256 else (128, 256)
    tiles_q = (sq + q_block - 1) // q_block
    tiles_kv = (skv + k_block - 1) // k_block

    if causal:
        mask = torch.ones(sq, skv, dtype=torch.bool).triu(diagonal=1 + (skv - sq))
    else:
        mask = torch.zeros(sq, skv, dtype=torch.bool)
    mask = mask.view(1, 1, sq, skv)

    min_value = -3.402823466e38
    out = torch.zeros(1, nq, sq, d)
    o_sum = torch.zeros(1, nq, sq, 1)
    o_max = torch.full((1, nq, sq, 1), min_value)

    for i in range(tiles_q):
        qs, qe = i * q_block, min((i + 1) * q_block, sq)
        qi = q[:, :, qs:qe] * dq[:, :, qs:qe]
        for j in range(tiles_kv):
            ks, ke = j * k_block, min((j + 1) * k_block, skv)
            kj = k[:, :, ks:ke] * dk[:, :, ks:ke]
            s_ij = torch.matmul(qi, kj.transpose(-1, -2)) * softmax_scale
            s_ij = s_ij.masked_fill(mask[:, :, qs:qe, ks:ke], float("-inf"))

            mi = o_max[:, :, qs:qe]
            m_block = s_ij.amax(dim=-1, keepdim=True)
            m_block = torch.ceil(m_block / ln2) * ln2  # NPU aligns max to ln2
            m_block = torch.max(mi, m_block)

            p_raw = torch.exp(s_ij - m_block)
            s_block = p_raw.sum(dim=-1, keepdim=True)
            p_drop = p_raw.to(FP8).float()  # P is stored as FP8_E4M3

            vj = v[:, :, ks:ke] * dv[:, :, ks:ke]
            update = torch.exp(mi - m_block)
            out[:, :, qs:qe] = update * out[:, :, qs:qe] + torch.matmul(p_drop, vj)
            o_sum[:, :, qs:qe] = update * o_sum[:, :, qs:qe] + s_block
            o_max[:, :, qs:qe] = m_block

    out = out / (o_sum + 1e-20)
    out = torch.where(o_max <= min_value, torch.zeros_like(out), out)
    return out.squeeze(0).permute(1, 0, 2)  # (Sq, Nq, D)


def compare(name: str, npu_out: torch.Tensor, golden: torch.Tensor) -> bool:
    """Two-per-mille family criterion, mirrored from result_compare_method."""
    npu = npu_out.float().cpu().reshape(-1)
    ref = golden.float().cpu().reshape(-1)
    abs_diff = (npu - ref).abs()
    rel_diff = abs_diff / ref.abs().clamp(min=1.0)
    ok = (abs_diff <= 1e-3) | (rel_diff <= 0.0078125)
    pass_rate = ok.float().mean().item()
    cos = torch.nn.functional.cosine_similarity(npu, ref, dim=0).item()
    max_abs = abs_diff.max().item()
    print(
        f"  [{name}] pass_rate={pass_rate:.6f} cos={cos:.6f} "
        f"max_abs_diff={max_abs:.6f} npu_mean={npu.mean():.6f} ref_mean={ref.mean():.6f}"
    )
    good = pass_rate >= 0.995 and cos >= 0.999
    print(f"  [{name}] {'GREEN' if good else 'RED'}")
    return good


# --------------------------------------------------------------------------
# NPU harness
# --------------------------------------------------------------------------
def bootstrap_ops():
    import torch_npu  # noqa: F401

    try:
        from vllm_ascend.utils import bootstrap_custom_op_env

        bootstrap_custom_op_env(include_vendor_lib=True)
    except Exception:  # fall back: derive vendor path from the package location
        import vllm_ascend

        vendor = os.path.join(
            os.path.dirname(vllm_ascend.__file__), "_cann_ops_custom", "vendors", "custom_transformer"
        )
        if os.path.isdir(vendor):
            prev = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = vendor + (":" + prev if prev else "")
        else:
            print(f"[WARN] custom opp vendor dir missing: {vendor}")
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    assert hasattr(torch.ops._C_ascend, "npu_quant_flash_attn"), "npu_quant_flash_attn not registered"
    assert hasattr(torch.ops._C_ascend, "npu_quant_flash_attn_metadata"), "metadata op not registered"
    print("[OK] torch.ops._C_ascend.npu_quant_flash_attn{,_metadata} registered")


def causal_mask_npu() -> torch.Tensor:
    return torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()


def call_qfa(npu_kwargs: dict, num_heads_q: int, num_heads_kv: int, head_dim: int, mask: torch.Tensor):
    """metadata + main op with strictly matching arguments."""
    common = {
        k: npu_kwargs.get(k)
        for k in (
            "cu_seqlens_q",
            "cu_seqlens_kv",
            "seqused_q",
            "seqused_kv",
            "mask_mode",
            "max_seqlen_q",
            "max_seqlen_kv",
            "layout_q",
            "layout_q_descale",
            "layout_kv",
            "layout_out",
        )
        if npu_kwargs.get(k) is not None
    }
    metadata = torch.ops._C_ascend.npu_quant_flash_attn_metadata(
        num_heads_q, num_heads_kv, head_dim, v_descale=npu_kwargs["v_descale"], **common
    )
    return torch.ops._C_ascend.npu_quant_flash_attn(
        npu_kwargs["q"],
        npu_kwargs["k"],
        npu_kwargs["v"],
        npu_kwargs["q_descale"],
        npu_kwargs["k_descale"],
        npu_kwargs["v_descale"],
        metadata,
        npu_kwargs["softmax_scale"],
        block_table=npu_kwargs.get("block_table"),
        attn_mask=mask,
        **common,
    )


def e8m0_npu(scale_bytes: torch.Tensor) -> torch.Tensor:
    return scale_bytes.npu().view(torch.float8_e8m0fnu)


# --------------------------------------------------------------------------
# Case 1: TND varlen prefill
# --------------------------------------------------------------------------
def case_tnd() -> bool:
    print("== case 1: TND varlen prefill (B=2, S=[128, 200], Nq=8, Nkv=2, D=128) ==")
    torch.manual_seed(1024)
    seq_lens, nq, nkv, d = [128, 200], 8, 2, 128
    softmax_scale = 1.0 / math.sqrt(d)

    q_chunks, k_chunks, v_chunks = [], [], []
    qs_chunks, ks_chunks, vs_chunks = [], [], []
    goldens = []
    for s in seq_lens:
        q = torch.randn(s, nq, d, dtype=torch.bfloat16)
        k = torch.randn(s, nkv, d, dtype=torch.bfloat16)
        v = torch.randn(s, nkv, d, dtype=torch.bfloat16)
        qsc, ksc = qk_group_scale(q), qk_group_scale(k)
        s_pad = (s + 63) // 64 * 64
        v_pad = torch.nn.functional.pad(v.float(), (0, 0, 0, 0, 0, s_pad - s))
        vsc = v_group_scale(v_pad)  # (s_pad//32, nkv, d)

        q_fp8 = quantize_with_scale(q, qsc.repeat_interleave(GROUP, dim=-1)[..., :d])
        k_fp8 = quantize_with_scale(k, ksc.repeat_interleave(GROUP, dim=-1)[..., :d])
        # expand v scale (Sg, N, D) along S for element-wise quant
        v_fp8 = quantize_with_scale(v_pad, vsc.repeat_interleave(GROUP, dim=0)[:s_pad])[:s]

        goldens.append(
            cpu_golden_one_seq(q_fp8, k_fp8, v_fp8, qsc, ksc, vsc[: (s + GROUP - 1) // GROUP], softmax_scale, True)
        )
        q_chunks.append(q_fp8)
        k_chunks.append(k_fp8)
        v_chunks.append(v_fp8)
        qs_chunks.append(pack_last_pairs(fp32_to_e8m0_bytes(qsc, "q_scale")))
        ks_chunks.append(pack_last_pairs(fp32_to_e8m0_bytes(ksc, "k_scale")))
        vs_chunks.append(pack_v_scale_seq(fp32_to_e8m0_bytes(vsc, "v_scale")))

    cu = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32)
    npu_kwargs = {
        "q": torch.cat(q_chunks).npu(),
        "k": torch.cat(k_chunks).npu(),
        "v": torch.cat(v_chunks).npu(),
        "q_descale": e8m0_npu(torch.cat(qs_chunks)),
        "k_descale": e8m0_npu(torch.cat(ks_chunks)),
        "v_descale": e8m0_npu(torch.cat(vs_chunks)),
        "cu_seqlens_q": cu.npu(),
        "cu_seqlens_kv": cu.npu(),
        "softmax_scale": softmax_scale,
        "mask_mode": 3,
        "max_seqlen_q": max(seq_lens),
        "max_seqlen_kv": max(seq_lens),
        "layout_q": "TND",
        "layout_q_descale": "TND",
        "layout_kv": "TND",
        "layout_out": "TND",
    }
    out = call_qfa(npu_kwargs, nq, nkv, d, causal_mask_npu())
    print(f"  attn_out: shape={tuple(out.shape)} dtype={out.dtype}")
    return compare("TND", out, torch.cat(goldens))


# --------------------------------------------------------------------------
# Case 2: PA_BBND decode
# --------------------------------------------------------------------------
def case_pa_bbnd() -> bool:
    print("== case 2: PA_BBND decode (B=2, Skv=[300, 257], block=128, Nq=8, Nkv=2, D=128) ==")
    torch.manual_seed(2048)
    kv_lens, nq, nkv, d, bs = [300, 257], 8, 2, 128, 128
    b = len(kv_lens)
    g = nq // nkv
    softmax_scale = 1.0 / math.sqrt(d)

    blocks_per_seq = [(s + bs - 1) // bs for s in kv_lens]
    total_blocks = sum(blocks_per_seq)
    k_cache = torch.zeros(total_blocks, bs, nkv, d, dtype=torch.uint8)
    v_cache = torch.zeros(total_blocks, bs, nkv, d, dtype=torch.uint8)
    k_scale_cache = torch.zeros(total_blocks, bs, nkv, d // 64, 2, dtype=torch.uint8)
    v_scale_cache = torch.full((total_blocks, bs // 64, nkv, d, 2), 127, dtype=torch.uint8)  # 2^0
    block_table = torch.zeros(b, max(blocks_per_seq), dtype=torch.int32)

    q_all, qs_all, goldens = [], [], []
    next_block = 0
    for i, s in enumerate(kv_lens):
        q = torch.randn(1, nq, d, dtype=torch.bfloat16)
        k = torch.randn(s, nkv, d, dtype=torch.bfloat16)
        v = torch.randn(s, nkv, d, dtype=torch.bfloat16)
        qsc, ksc = qk_group_scale(q), qk_group_scale(k)
        s_pad = blocks_per_seq[i] * bs
        v_pad = torch.nn.functional.pad(v.float(), (0, 0, 0, 0, 0, s_pad - s))
        vsc = v_group_scale(v_pad)  # (s_pad//32, nkv, d)

        q_fp8 = quantize_with_scale(q, qsc.repeat_interleave(GROUP, dim=-1)[..., :d])
        k_fp8 = quantize_with_scale(k, ksc.repeat_interleave(GROUP, dim=-1)[..., :d])
        v_fp8 = quantize_with_scale(v_pad, vsc.repeat_interleave(GROUP, dim=0)[:s_pad])[:s]
        goldens.append(
            cpu_golden_one_seq(q_fp8, k_fp8, v_fp8, qsc, ksc, vsc[: (s + GROUP - 1) // GROUP], softmax_scale, True)
        )
        q_all.append(q_fp8)
        qs_all.append(fp32_to_e8m0_bytes(qsc, "q_scale"))

        k_bytes = torch.nn.functional.pad(k_fp8.view(torch.uint8), (0, 0, 0, 0, 0, s_pad - s))
        v_bytes = torch.nn.functional.pad(v_fp8.view(torch.uint8), (0, 0, 0, 0, 0, s_pad - s))
        ks_bytes = pack_last_pairs(fp32_to_e8m0_bytes(ksc, "k_scale"))
        ks_bytes = torch.nn.functional.pad(ks_bytes, (0, 0, 0, 0, 0, 0, 0, s_pad - s))
        vs_bytes = pack_v_scale_seq(fp32_to_e8m0_bytes(vsc, "v_scale"))  # (s_pad//64, nkv, d, 2)
        for j in range(blocks_per_seq[i]):
            blk = next_block
            next_block += 1
            block_table[i, j] = blk
            k_cache[blk] = k_bytes[j * bs : (j + 1) * bs]
            v_cache[blk] = v_bytes[j * bs : (j + 1) * bs]
            k_scale_cache[blk] = ks_bytes[j * bs : (j + 1) * bs]
            v_scale_cache[blk] = vs_bytes[j * (bs // 64) : (j + 1) * (bs // 64)]

    # q_descale N2TGD: (Nkv, Tq, G, D//64, 2) selects the decode tiling template
    qs = pack_last_pairs(torch.cat(qs_all))  # (Tq, Nq, D//64, 2)
    qs_n2tgd = qs.reshape(b, nkv, g, d // 64, 2).permute(1, 0, 2, 3, 4).contiguous()

    npu_kwargs = {
        "q": torch.cat(q_all).npu(),
        "k": k_cache.npu().view(torch.float8_e4m3fn),
        "v": v_cache.npu().view(torch.float8_e4m3fn),
        "q_descale": e8m0_npu(qs_n2tgd),
        "k_descale": e8m0_npu(k_scale_cache),
        "v_descale": e8m0_npu(v_scale_cache),
        "block_table": block_table.npu(),
        "cu_seqlens_q": torch.arange(b + 1, dtype=torch.int32).npu(),
        "seqused_kv": torch.tensor(kv_lens, dtype=torch.int32).npu(),
        "softmax_scale": softmax_scale,
        "mask_mode": 3,
        "max_seqlen_q": 1,
        "max_seqlen_kv": max(kv_lens),
        "layout_q": "TND",
        "layout_q_descale": "N2TGD",
        "layout_kv": "PA_BBND",
        "layout_out": "TND",
    }
    out = call_qfa(npu_kwargs, nq, nkv, d, causal_mask_npu())
    print(f"  attn_out: shape={tuple(out.shape)} dtype={out.dtype}")
    return compare("PA_BBND", out, torch.cat(goldens))


# --------------------------------------------------------------------------
# Case 3: online-quant consistency (npu_dynamic_mx_quant vs CPU reference)
# --------------------------------------------------------------------------
def case_quant_consistency() -> bool:
    print("== case 3: npu_dynamic_mx_quant vs CPU reference (incl. all-zero group) ==")
    import torch_npu

    torch.manual_seed(4096)
    t, d = 64, 128
    x = torch.randn(t, d, dtype=torch.bfloat16)
    x[0, :GROUP] = 0  # one all-zero group
    x[-1] = 0  # one all-zero row (pure padding scenario)

    ref_scale = qk_group_scale(x.unsqueeze(1)).squeeze(1)  # (T, D//32)
    ref_fp8 = quantize_with_scale(x.unsqueeze(1), ref_scale.unsqueeze(1).repeat_interleave(GROUP, dim=-1)[..., :d])
    ref_fp8 = ref_fp8.squeeze(1)
    zero_group = torch.all(x.float().reshape(t, d // GROUP, GROUP) == 0, dim=-1)  # (T, D//32)

    fp8, scale = torch_npu.npu_dynamic_mx_quant(x.npu(), dst_type=torch.float8_e4m3fn, scale_alg=0)
    scale_bytes = scale.view(torch.uint8).cpu().reshape(t, d // GROUP)
    nan_bytes = int((scale_bytes == 0xFF).sum())
    ref_bytes = fp32_to_e8m0_bytes(ref_scale, "ref_scale")
    zero_group_bytes = sorted(set(scale_bytes[zero_group].tolist()))
    print(
        f"  npu scale: shape={tuple(scale.shape)} nan_bytes(0xFF)={nan_bytes} "
        f"all-zero-group bytes={zero_group_bytes} (any non-0xFF value is safe)"
    )

    # All-zero groups have no canonical scale; compare only meaningful groups.
    nonzero = ~zero_group
    scale_match = (scale_bytes[nonzero] == ref_bytes[nonzero]).float().mean().item()
    fp8_match = (fp8.cpu().view(torch.uint8) == ref_fp8.view(torch.uint8)).float().mean().item()
    print(f"  scale byte match={scale_match:.6f} (non-zero groups) fp8 byte match={fp8_match:.6f}")
    # Exact scale agreement is expected for alg 0; fp8 rounding may differ on ties.
    good = nan_bytes == 0 and scale_match >= 0.999 and fp8_match >= 0.99
    print(f"  [QUANT] {'GREEN' if good else 'RED'}")
    return good


# --------------------------------------------------------------------------
# Case 4: Qwen3.8-27B prefill shape repro (crash isolation, staged syncs)
# --------------------------------------------------------------------------
def _online_quant_qk(x: torch.Tensor):
    """Mirror of AscendAttentionBackendImpl._qfa_quant_query_key (scale_alg=0)."""
    import torch_npu

    t, n, d = x.shape
    fp8, scale = torch_npu.npu_dynamic_mx_quant(
        x.reshape(t * n, d), dst_type=torch.float8_e4m3fn, scale_alg=0
    )
    scale = scale.view(torch.uint8).reshape(t, n, d // 64, 2)
    return fp8.reshape(t, n, d), scale.view(torch.float8_e8m0fnu)


def _online_quant_v(v: torch.Tensor, s: int):
    """Mirror of _qfa_quant_value_per_seq for a single sequence."""
    import torch_npu

    n, d = v.shape[1], v.shape[2]
    padded = (s + 63) // 64 * 64
    chunk = v
    if padded != s:
        chunk = torch.nn.functional.pad(chunk, (0, 0, 0, 0, 0, padded - s))
    chunk = chunk.permute(1, 2, 0).contiguous().view(n * d, padded)
    fp8, scale = torch_npu.npu_dynamic_mx_quant(chunk, dst_type=torch.float8_e4m3fn, scale_alg=0)
    fp8 = fp8.view(n, d, padded).permute(2, 0, 1)[:s].contiguous()
    scale = scale.view(torch.uint8).reshape(n, d, padded // 64, 2).permute(2, 0, 1, 3).contiguous()
    return fp8, scale.view(torch.float8_e8m0fnu)


def case_27b_shape() -> bool:
    s = int(os.environ.get("QFA27B_S", "1594"))
    nq = int(os.environ.get("QFA27B_NQ", "24"))
    nkv = int(os.environ.get("QFA27B_NKV", "4"))
    d = int(os.environ.get("QFA27B_D", "256"))
    with_golden = os.environ.get("QFA27B_GOLDEN", "1") == "1"
    print(f"== case 4: 27B prefill shape repro (B=1, S={s}, Nq={nq}, Nkv={nkv}, D={d}) ==")
    torch.manual_seed(27)
    softmax_scale = 1.0 / math.sqrt(d)

    q = torch.randn(s, nq, d, dtype=torch.bfloat16)
    k = torch.randn(s, nkv, d, dtype=torch.bfloat16)
    v = torch.randn(s, nkv, d, dtype=torch.bfloat16)
    qsc, ksc = qk_group_scale(q), qk_group_scale(k)
    s_pad = (s + 63) // 64 * 64
    v_pad = torch.nn.functional.pad(v.float(), (0, 0, 0, 0, 0, s_pad - s))
    vsc = v_group_scale(v_pad)  # (s_pad//32, nkv, d)

    q_fp8 = quantize_with_scale(q, qsc.repeat_interleave(GROUP, dim=-1)[..., :d])
    k_fp8 = quantize_with_scale(k, ksc.repeat_interleave(GROUP, dim=-1)[..., :d])
    v_fp8 = quantize_with_scale(v_pad, vsc.repeat_interleave(GROUP, dim=0)[:s_pad])[:s]

    cu = torch.tensor([0, s], dtype=torch.int32).npu()
    common = {
        "cu_seqlens_q": cu,
        "cu_seqlens_kv": cu,
        "mask_mode": 3,
        "max_seqlen_q": s,
        "max_seqlen_kv": s,
        "layout_q": "TND",
        "layout_q_descale": "TND",
        "layout_kv": "TND",
        "layout_out": "TND",
    }
    q_npu = q_fp8.npu()
    k_npu = k_fp8.npu()
    v_npu = v_fp8.npu()
    qs_npu = e8m0_npu(pack_last_pairs(fp32_to_e8m0_bytes(qsc, "q_scale")))
    ks_npu = e8m0_npu(pack_last_pairs(fp32_to_e8m0_bytes(ksc, "k_scale")))
    vs_npu = e8m0_npu(pack_v_scale_seq(fp32_to_e8m0_bytes(vsc, "v_scale")))
    mask = causal_mask_npu()
    torch.npu.synchronize()  # inputs materialized; failures past here are the op's

    # Stage 1: metadata AICPU op alone. The 27B server crash reported
    # kernelName=QuantFlashAttnMetadata, so a hang/abort here confirms it.
    print("  [stage] metadata AICPU op ...", flush=True)
    metadata = torch.ops._C_ascend.npu_quant_flash_attn_metadata(
        nq, nkv, d, v_descale=vs_npu, **common
    )
    torch.npu.synchronize()
    head = metadata.cpu()[:4].tolist()
    print(
        f"  [stage] metadata OK: sectionNum={head[0]} isFd={head[1]} "
        f"mBase={head[2]} s2Base={head[3]}",
        flush=True,
    )

    # Stage 2: main op with the very same argument set.
    print("  [stage] main op ...", flush=True)
    out = torch.ops._C_ascend.npu_quant_flash_attn(
        q_npu, k_npu, v_npu, qs_npu, ks_npu, vs_npu, metadata, softmax_scale,
        attn_mask=mask, **common,
    )
    torch.npu.synchronize()
    print(f"  [stage] main op OK: shape={tuple(out.shape)} dtype={out.dtype}", flush=True)

    # Stage 3 (optional): back-to-back layers like a real 27B forward pass.
    # The server launches online-quant + metadata + main for every full-attn
    # layer with no host sync in between, and each layer's temporaries are
    # freed and recycled by the caching allocator while earlier AICPU tasks
    # may still sit in the stream queue. QFA27B_ITERS=12 reproduces exactly
    # that queue shape (also the first on-device run of online npu_dynamic_
    # mx_quant feeding QFA, D=256 included).
    iters = int(os.environ.get("QFA27B_ITERS", "1"))
    if iters > 1:
        q_bf, k_bf, v_bf = q.npu(), k.npu(), v.npu()
        torch.npu.synchronize()
        print(
            f"  [stage] back-to-back x{iters} (online-quant+metadata+main, no host sync) ...",
            flush=True,
        )
        last = None
        for _ in range(iters):
            qf, qd = _online_quant_qk(q_bf)
            kf, kd = _online_quant_qk(k_bf)
            vf, vd = _online_quant_v(v_bf, s)
            md = torch.ops._C_ascend.npu_quant_flash_attn_metadata(
                nq, nkv, d, v_descale=vd, **common
            )
            last = torch.ops._C_ascend.npu_quant_flash_attn(
                qf, kf, vf, qd, kd, vd, md, softmax_scale, attn_mask=mask, **common
            )
        torch.npu.synchronize()
        print(f"  [stage] back-to-back OK: shape={tuple(last.shape)}", flush=True)

    if not with_golden:
        print("  [27B] GREEN (no-crash mode, golden skipped via QFA27B_GOLDEN=0)")
        return True
    print("  [stage] CPU golden (may take a minute or two) ...", flush=True)
    golden = cpu_golden_one_seq(
        q_fp8, k_fp8, v_fp8, qsc, ksc, vsc[: (s + GROUP - 1) // GROUP], softmax_scale, True
    )
    return compare("27B", out, golden)


def main() -> int:
    global FP8
    FP8 = torch.float8_e4m3fn
    device_id = int(os.environ.get("QFA_TEST_DEVICE", "0"))
    import torch_npu  # noqa: F401

    torch.npu.set_device(device_id)
    print(f"[INFO] device npu:{device_id}")
    bootstrap_ops()

    all_cases = (
        ("TND", case_tnd),
        ("PA_BBND", case_pa_bbnd),
        ("QUANT", case_quant_consistency),
        ("27B", case_27b_shape),
    )
    only = os.environ.get("QFA_CASES")
    if only:
        wanted = {token.strip() for token in only.split(",") if token.strip()}
        cases = tuple(case for case in all_cases if case[0] in wanted)
        if not cases:
            print(f"[ERROR] QFA_CASES={only!r} matches none of {[c[0] for c in all_cases]}")
            return 2
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
    all_ok = all(results.values())
    print(f"[{'GREEN' if all_ok else 'RED'}] quant_flash_attn NPU smoke overall")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
