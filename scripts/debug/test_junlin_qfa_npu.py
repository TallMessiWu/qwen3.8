#!/usr/bin/env python3
"""Server-side numeric smoke test for the vendored official quant_flash_attn.

junlin-qfa branch: the official ops-transformer master QuantFlashAttn (MXFP8,
quant_mode=1) is vendored into vllm-ascend csrc and exposed as
torch.ops._C_ascend.npu_quant_flash_attn{,_metadata}. This script validates
numerics on a single A5 NPU against a CPU golden ported from the official test
assets (attention/quant_flash_attn/tests/assets/quant_flash_attn_golden.py
@ ops-transformer 14cf794f3): same MXFP8 group quantization, same e8m0 scale
packing, same C1V1C1V1C2V2 blockwise pipeline with ln2-aligned running max.

  case 1 TND-SMALL  varlen prefill B=2 S=[128, 200]  Nq=8  Nkv=2 D=128
  case 2 TND-27B    varlen prefill B=2 S=[512, 300]  Nq=24 Nkv=4 D=256
                    (Qwen3.8-27B per-rank head shape)
  case 3 PA-TND     paged decode q=1 B=2 kv=[300, 257] block=128,
                    layout_kv=PA_BNBD, q_descale layout TND
  case 4 PA-N2TGD   paged decode q=1 27B shape kv=[1, 127, 128, 129, 300]
                    block=128, q_descale layout N2TGD (decode tiling template)

Every case calls npu_quant_flash_attn_metadata first and the main op second
with an identical argument set (the op does no cross-call validation).
Prints [GREEN]/[RED] per case and exits non-zero on any RED.

Usage (inside the serving container, after pip_install_qfa.sh):
  python scripts/debug/test_junlin_qfa_npu.py
  QFA_CASES=PA-N2TGD python scripts/debug/test_junlin_qfa_npu.py   # one case
"""

import math
import os
import sys
import traceback

import torch

GROUP = 32
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
EMAX_E4M3 = 8
E8M0_MIN_POSITIVE = 2.0 ** (-127)
LN2 = 0.6931471824645996
INV_LN2 = 1.4426950216293335
MIN_VALUE = -3.402823466e38


# --------------------------------------------------------------------------
# MXFP8 quantization (ported from get_mxfp8_per_*_group_quant_scale)
# --------------------------------------------------------------------------
def qk_group_scale_bnsd(x: torch.Tensor) -> torch.Tensor:
    """Per-token-group scale over D for (B, N, S, D); returns (B, N, S, Dg)
    with Dg = ceil64(D) / 32 (D padded to a multiple of 64 first)."""
    b, n, s, d = x.shape
    d_align = (d + 63) // 64 * 64
    num_groups = d_align // GROUP
    pad = num_groups * GROUP - d
    xf = x.float()
    if pad:
        xf = torch.nn.functional.pad(xf, (0, pad))
    grouped = xf.reshape(b, n, s, num_groups, GROUP)
    all_zero = torch.all(grouped == 0, dim=-1)
    max_vals = grouped.abs().amax(dim=-1).clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - EMAX_E4M3
    return torch.where(all_zero, torch.ones_like(shared_exp), 2.0**shared_exp)


def v_group_scale_bnsd(x: torch.Tensor) -> torch.Tensor:
    """Per-channel-group scale over S for (B, N, S, D); returns (B, N, Sg, D)
    with Sg = ceil(S / 32)."""
    b, n, s, d = x.shape
    num_groups = (s + GROUP - 1) // GROUP
    pad = num_groups * GROUP - s
    xf = x.float()
    if pad:
        xf = torch.nn.functional.pad(xf, (0, 0, 0, pad))
    grouped = xf.reshape(b, n, num_groups, GROUP, d)
    all_zero = torch.all(grouped == 0, dim=-2)
    max_vals = grouped.abs().amax(dim=-2).clamp(min=1e-12)
    shared_exp = torch.floor(torch.log2(max_vals)) - EMAX_E4M3
    return torch.where(all_zero, torch.ones_like(shared_exp), 2.0**shared_exp)


def quant_qk_bnsd(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    expanded = scale.repeat_interleave(GROUP, dim=-1)[..., :d]
    return (x.float() / expanded).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)


def quant_v_bnsd(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    s = x.shape[2]
    expanded = scale.repeat_interleave(GROUP, dim=2)[:, :, :s, :]
    return (x.float() / expanded).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)


# --------------------------------------------------------------------------
# e8m0 conversion + NPU scale packing (fp32_to_e8m0fnu / pack_*_for_npu)
# --------------------------------------------------------------------------
def fp32_to_e8m0_bytes(scale: torch.Tensor, name: str) -> torch.Tensor:
    safe = scale.float().clone()
    safe[~torch.isfinite(safe)] = E8M0_MIN_POSITIVE
    safe[safe == 0] = E8M0_MIN_POSITIVE
    bits = safe.view(torch.int32)
    exp = ((bits >> 23) & 0xFF).to(torch.uint8)
    nan_count = int((exp == 0xFF).sum())
    if nan_count:
        raise ValueError(f"{name}: {nan_count} scale values would become e8m0 NaN")
    return exp


def pack_qk_scale(scale_bytes: torch.Tensor) -> torch.Tensor:
    """(..., Dg) uint8 -> (..., Dg//2, 2): adjacent groups pack pairwise."""
    shape = scale_bytes.shape
    return scale_bytes.reshape(*shape[:-1], shape[-1] // 2, 2)


def pack_v_scale_rows(scale_sg_n_d: torch.Tensor) -> torch.Tensor:
    """(Sg, N, D) uint8 -> (ceil(Sg/2), N, D, 2): even rows -> [..., 0],
    odd rows -> [..., 1]; odd Sg pads one row with the 2^-127 byte (0x00)."""
    sg, n, d = scale_sg_n_d.shape
    if sg % 2:
        pad = torch.zeros((1, n, d), dtype=torch.uint8)  # 0x00 == 2^-127
        scale_sg_n_d = torch.cat([scale_sg_n_d, pad], dim=0)
        sg += 1
    out = torch.zeros((sg // 2, n, d, 2), dtype=torch.uint8)
    out[..., 0] = scale_sg_n_d[::2]
    out[..., 1] = scale_sg_n_d[1::2]
    return out


def q_scale_tnd_to_n2tgd(packed_tnd: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """(T, Nq, Dg//2, 2) -> (Nkv, T, G, Dg//2, 2)."""
    t, n, dgh, two = packed_tnd.shape
    g = n // num_kv_heads
    return packed_tnd.reshape(t, num_kv_heads, g, dgh, two).permute(1, 0, 2, 3, 4).contiguous()


def e8m0_npu(scale_bytes: torch.Tensor) -> torch.Tensor:
    return scale_bytes.npu().view(torch.float8_e8m0fnu)


# --------------------------------------------------------------------------
# CPU golden (ported from cpu_mxfp8_golden: C1V1C1V1C2V2 pipeline)
# --------------------------------------------------------------------------
def _build_attention_mask(sq, skv, actual_q, actual_kv, mask_mode):
    """(1, 1, Sq, Skv) bool for one sequence; sparse 3 = bottom-right causal."""
    q_range = torch.arange(sq).view(1, 1, -1, 1)
    k_range = torch.arange(skv).view(1, 1, 1, -1)
    q_pad = q_range >= actual_q
    k_pad = k_range >= actual_kv
    if mask_mode == 3:
        delta = actual_kv - actual_q
        causal = k_range > (q_range + delta)
        return causal | q_pad | k_pad
    return q_pad | k_pad


def _online_softmax_update(s_ij, mask_j, mi, ln_p_scale):
    s_ij = s_ij.masked_fill(mask_j, float("-inf"))
    m_block, _ = torch.max(s_ij, dim=-1, keepdims=True)
    m_block = torch.ceil(m_block * INV_LN2) * LN2
    m_block = torch.max(mi, m_block)
    p_raw = torch.exp(s_ij - (m_block - ln_p_scale))
    s_block = torch.sum(p_raw, dim=-1, keepdims=True)
    p_drop = p_raw.to(FP8_DTYPE).to(torch.float32)
    return m_block, s_block, p_drop


def cpu_golden_one_seq(
    q_fp8,  # (Sq, Nq, D) fp8
    k_fp8,  # (Skv, Nkv, D) fp8
    v_fp8,  # (Skv, Nkv, D) fp8
    q_scale,  # (1, Nq, Sq, Dg) fp32 group granularity
    k_scale,  # (1, Nkv, Skv, Dg) fp32
    v_scale,  # (1, Nkv, Sg, D) fp32
    softmax_scale: float,
    mask_mode: int,
) -> torch.Tensor:
    """Single sequence, BNSD internally, returns (Sq, Nq, D) fp32."""
    sq, nq, d = q_fp8.shape
    skv, nkv, _ = k_fp8.shape
    group_rep = nq // nkv

    def to_bnsd(x):  # (S, N, D) -> (1, N, S, D) fp32
        return x.float().permute(1, 0, 2).unsqueeze(0)

    q = to_bnsd(q_fp8)
    k = to_bnsd(k_fp8).repeat_interleave(group_rep, dim=1)
    v = to_bnsd(v_fp8).repeat_interleave(group_rep, dim=1)
    dq = q_scale.repeat_interleave(GROUP, dim=-1)[..., :d]
    dk = k_scale.repeat_interleave(group_rep, dim=1).repeat_interleave(GROUP, dim=-1)[..., :d]
    dv = v_scale.repeat_interleave(group_rep, dim=1).repeat_interleave(GROUP, dim=2)[:, :, :skv, :]

    q_block = 128
    if d == 256:
        k_block, v_block = 128, 256
    else:
        k_block, v_block = 256, 512

    mask_global = _build_attention_mask(sq, skv, sq, skv, mask_mode)

    tiles_q = (sq + q_block - 1) // q_block
    tiles_kv = (skv + k_block - 1) // k_block

    out = torch.zeros(1, nq, sq, d)
    o_sum = torch.zeros(1, nq, sq, 1)
    o_max = torch.full((1, nq, sq, 1), MIN_VALUE)
    ln_p_scale = torch.tensor([0.0])  # p_scale defaults to 1.0

    for i in range(tiles_q):
        qs, qe = i * q_block, min((i + 1) * q_block, sq)
        qi = q[:, :, qs:qe] * dq[:, :, qs:qe]
        for j in range(0, tiles_kv, 2):
            oi = out[:, :, qs:qe]
            si = o_sum[:, :, qs:qe]
            mi = o_max[:, :, qs:qe]

            ks, ke = j * k_block, min((j + 1) * k_block, skv)
            kj = k[:, :, ks:ke] * dk[:, :, ks:ke]
            s_ij = torch.matmul(qi, kj.transpose(-1, -2)) * softmax_scale
            m_j, s_j, p_j = _online_softmax_update(
                s_ij, mask_global[:, :, qs:qe, ks:ke], mi, ln_p_scale
            )

            if j + 1 < tiles_kv:
                ks1, ke1 = (j + 1) * k_block, min((j + 2) * k_block, skv)
                kj1 = k[:, :, ks1:ke1] * dk[:, :, ks1:ke1]
                s_ij1 = torch.matmul(qi, kj1.transpose(-1, -2)) * softmax_scale
                m_j1, s_j1, p_j1 = _online_softmax_update(
                    s_ij1, mask_global[:, :, qs:qe, ks1:ke1], m_j, ln_p_scale
                )

                vs, ve = (j // 2) * v_block, min((j // 2 + 1) * v_block, skv)
                vj = v[:, :, vs:ve] * dv[:, :, vs:ve]
                v_part1 = vj[:, :, : ke - ks]
                v_part2 = vj[:, :, ke - ks : (ke - ks) + (ke1 - ks1)]
                pv = torch.matmul(p_j * torch.exp(m_j - m_j1), v_part1) + torch.matmul(
                    p_j1, v_part2
                )
                update = torch.exp(mi - m_j1)
                out[:, :, qs:qe] = update * oi + pv
                o_sum[:, :, qs:qe] = (
                    update * si + s_j * torch.exp(m_j - m_j1) + s_j1
                )
                o_max[:, :, qs:qe] = m_j1
            else:
                vs = j * k_block
                vj = v[:, :, vs : vs + (ke - ks)] * dv[:, :, vs : vs + (ke - ks)]
                pv = torch.matmul(p_j, vj)
                update = torch.exp(mi - m_j)
                out[:, :, qs:qe] = update * oi + pv
                o_sum[:, :, qs:qe] = update * si + s_j
                o_max[:, :, qs:qe] = m_j

    out = out / (o_sum + 1e-20)
    out = torch.where(o_max <= MIN_VALUE, torch.zeros_like(out), out)
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
    print(
        f"  [{name}] pass_rate={pass_rate:.6f} cos={cos:.6f} "
        f"max_abs_diff={abs_diff.max().item():.6f} "
        f"npu_mean={npu.mean():.6f} ref_mean={ref.mean():.6f}"
    )
    good = pass_rate >= 0.995 and cos >= 0.999
    print(f"  [{name}] {'GREEN' if good else 'RED'}")
    return good


# --------------------------------------------------------------------------
# Per-sequence quantization bundle
# --------------------------------------------------------------------------
class QuantSeq:
    """Quantize one sequence of bf16 q/k/v (S, N, D) into fp8 + scales."""

    def __init__(self, q, k, v):
        def to_bnsd(x):
            return x.float().permute(1, 0, 2).unsqueeze(0)

        self.q_scale = qk_group_scale_bnsd(to_bnsd(q))  # (1, Nq, Sq, Dg)
        self.k_scale = qk_group_scale_bnsd(to_bnsd(k))
        self.v_scale = v_group_scale_bnsd(to_bnsd(v))  # (1, Nkv, Sg, D)
        self.q_fp8 = (
            quant_qk_bnsd(to_bnsd(q), self.q_scale).squeeze(0).permute(1, 0, 2)
        )  # (Sq, Nq, D) fp8
        self.k_fp8 = quant_qk_bnsd(to_bnsd(k), self.k_scale).squeeze(0).permute(1, 0, 2)
        self.v_fp8 = quant_v_bnsd(to_bnsd(v), self.v_scale).squeeze(0).permute(1, 0, 2)

    def qk_scale_packed_tnd(self, which: str) -> torch.Tensor:
        """(S, N, Dg//2, 2) uint8 for q or k."""
        scale = self.q_scale if which == "q" else self.k_scale
        tnd = scale.squeeze(0).permute(1, 0, 2)  # (S, N, Dg)
        return pack_qk_scale(fp32_to_e8m0_bytes(tnd, which + "_scale"))

    def v_scale_packed_rows(self) -> torch.Tensor:
        """(ceil(Sg/2), Nkv, D, 2) uint8."""
        sg_n_d = self.v_scale.squeeze(0).permute(1, 0, 2)  # (Sg, Nkv, D)
        return pack_v_scale_rows(fp32_to_e8m0_bytes(sg_n_d, "v_scale"))

    def golden(self, softmax_scale, mask_mode):
        return cpu_golden_one_seq(
            self.q_fp8, self.k_fp8, self.v_fp8,
            self.q_scale, self.k_scale, self.v_scale,
            softmax_scale, mask_mode,
        )


# --------------------------------------------------------------------------
# NPU harness
# --------------------------------------------------------------------------
def bootstrap_ops():
    import torch_npu  # noqa: F401

    try:
        from vllm_ascend.utils import bootstrap_custom_op_env

        bootstrap_custom_op_env(include_vendor_lib=True)
    except Exception:
        import vllm_ascend

        vendor = os.path.join(
            os.path.dirname(vllm_ascend.__file__), "_cann_ops_custom", "vendors",
            "custom_transformer",
        )
        if os.path.isdir(vendor):
            prev = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = vendor + (":" + prev if prev else "")
        else:
            print(f"[WARN] custom opp vendor dir missing: {vendor}")
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    assert hasattr(torch.ops._C_ascend, "npu_quant_flash_attn"), \
        "npu_quant_flash_attn not registered"
    assert hasattr(torch.ops._C_ascend, "npu_quant_flash_attn_metadata"), \
        "npu_quant_flash_attn_metadata not registered"
    print("[OK] torch.ops._C_ascend.npu_quant_flash_attn{,_metadata} registered")


def causal_mask_npu() -> torch.Tensor:
    # Official test assets use triu(diagonal=1) int8 (1 = masked future);
    # the tril in the doc example contradicts the shipped tests.
    return torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()


def call_qfa(kwargs: dict, num_heads_q: int, num_heads_kv: int, head_dim: int):
    """Metadata + main op with a strictly matching argument set."""
    common_keys = (
        "cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv",
        "mask_mode", "max_seqlen_q", "max_seqlen_kv",
        "layout_q", "layout_q_descale", "layout_kv", "layout_out",
    )
    common = {k: kwargs[k] for k in common_keys if kwargs.get(k) is not None}
    metadata = torch.ops._C_ascend.npu_quant_flash_attn_metadata(
        num_heads_q, num_heads_kv, head_dim, 1,
        v_descale=kwargs["v_descale"], **common,
    )
    attn_out, _ = torch.ops._C_ascend.npu_quant_flash_attn(
        kwargs["q"], kwargs["k"], kwargs["v"],
        kwargs["q_descale"], kwargs["k_descale"], kwargs["v_descale"], 1,
        block_table=kwargs.get("block_table"),
        attn_mask=kwargs.get("attn_mask"),
        metadata=metadata,
        softmax_scale=kwargs["softmax_scale"],
        **common,
    )
    return attn_out


# --------------------------------------------------------------------------
# Case runners
# --------------------------------------------------------------------------
def run_tnd_case(name, seq_lens, nq, nkv, d, seed) -> bool:
    print(f"== {name}: TND varlen prefill B={len(seq_lens)} S={seq_lens} "
          f"Nq={nq} Nkv={nkv} D={d} ==")
    torch.manual_seed(seed)
    softmax_scale = 1.0 / math.sqrt(d)

    quants = []
    for s in seq_lens:
        q = torch.randn(s, nq, d, dtype=torch.bfloat16)
        k = torch.randn(s, nkv, d, dtype=torch.bfloat16)
        v = torch.randn(s, nkv, d, dtype=torch.bfloat16)
        quants.append(QuantSeq(q, k, v))

    goldens = [qs.golden(softmax_scale, mask_mode=3) for qs in quants]

    cu = torch.tensor([0] + torch.tensor(seq_lens).cumsum(0).tolist(), dtype=torch.int32)
    kwargs = {
        "q": torch.cat([qs.q_fp8 for qs in quants]).npu(),
        "k": torch.cat([qs.k_fp8 for qs in quants]).npu(),
        "v": torch.cat([qs.v_fp8 for qs in quants]).npu(),
        "q_descale": e8m0_npu(torch.cat([qs.qk_scale_packed_tnd("q") for qs in quants])),
        "k_descale": e8m0_npu(torch.cat([qs.qk_scale_packed_tnd("k") for qs in quants])),
        "v_descale": e8m0_npu(torch.cat([qs.v_scale_packed_rows() for qs in quants])),
        "cu_seqlens_q": cu.npu(),
        "cu_seqlens_kv": cu.clone().npu(),
        "softmax_scale": softmax_scale,
        "mask_mode": 3,
        "max_seqlen_q": max(seq_lens),
        "max_seqlen_kv": max(seq_lens),
        "layout_q": "TND",
        "layout_q_descale": "TND",
        "layout_kv": "TND",
        "layout_out": "TND",
        "attn_mask": causal_mask_npu(),
    }
    out = call_qfa(kwargs, nq, nkv, d)
    torch.npu.synchronize()
    print(f"  attn_out: shape={tuple(out.shape)} dtype={out.dtype}")
    return compare(name, out, torch.cat(goldens))


def run_pa_bnbd_case(name, kv_lens, nq, nkv, d, block_size, q_descale_layout, seed) -> bool:
    print(f"== {name}: PA_BNBD decode q=1 B={len(kv_lens)} kv={kv_lens} "
          f"Nq={nq} Nkv={nkv} D={d} block={block_size} "
          f"q_descale={q_descale_layout} ==")
    torch.manual_seed(seed)
    b = len(kv_lens)
    softmax_scale = 1.0 / math.sqrt(d)
    dg = ((d + 63) // 64 * 64) // GROUP  # quant groups per token

    blocks_per_seq = [max(1, (s + block_size - 1) // block_size) for s in kv_lens]
    total_blocks = sum(blocks_per_seq)
    # PA_BNBD cache layout: (Bn, N, Bs, D); scales analogous.
    k_cache = torch.zeros(total_blocks, nkv, block_size, d, dtype=torch.uint8)
    v_cache = torch.zeros(total_blocks, nkv, block_size, d, dtype=torch.uint8)
    k_scale_cache = torch.zeros(total_blocks, nkv, block_size, dg // 2, 2, dtype=torch.uint8)
    v_scale_cache = torch.zeros(total_blocks, nkv, block_size // 64, d, 2, dtype=torch.uint8)
    block_table = torch.zeros(b, max(blocks_per_seq), dtype=torch.int32)

    q_all, q_scale_all, goldens = [], [], []
    next_block = 0
    for i, s in enumerate(kv_lens):
        q = torch.randn(1, nq, d, dtype=torch.bfloat16)
        if s == 0:
            qs0 = QuantSeq(q, torch.zeros(1, nkv, d, dtype=torch.bfloat16),
                           torch.zeros(1, nkv, d, dtype=torch.bfloat16))
            q_all.append(qs0.q_fp8)
            q_scale_all.append(qs0.qk_scale_packed_tnd("q"))
            goldens.append(torch.zeros(1, nq, d))
            block_table[i, 0] = next_block
            next_block += 1
            continue
        k = torch.randn(s, nkv, d, dtype=torch.bfloat16)
        v = torch.randn(s, nkv, d, dtype=torch.bfloat16)
        # Quantize K/V over the block-aligned padded length so every 32-token
        # V group lies inside one block and pad groups quantize to scale=1.
        s_pad = blocks_per_seq[i] * block_size
        k_pad = torch.nn.functional.pad(k.float(), (0, 0, 0, 0, 0, s_pad - s)).bfloat16()
        v_pad = torch.nn.functional.pad(v.float(), (0, 0, 0, 0, 0, s_pad - s)).bfloat16()
        qs = QuantSeq(q, k_pad, v_pad)
        q_all.append(qs.q_fp8)
        q_scale_all.append(qs.qk_scale_packed_tnd("q"))

        sg_real = (s + GROUP - 1) // GROUP
        goldens.append(
            cpu_golden_one_seq(
                qs.q_fp8, qs.k_fp8[:s], qs.v_fp8[:s],
                qs.q_scale, qs.k_scale[:, :, :s], qs.v_scale[:, :, :sg_real],
                softmax_scale, mask_mode=3,
            )
        )

        k_bytes = qs.k_fp8.view(torch.uint8)  # (s_pad, nkv, d)
        v_bytes = qs.v_fp8.view(torch.uint8)
        ks_bytes = qs.qk_scale_packed_tnd("k")  # (s_pad, nkv, dg//2, 2)
        vs_rows = qs.v_scale_packed_rows()  # (s_pad//64, nkv, d, 2)
        rows_per_block = block_size // 64
        for j in range(blocks_per_seq[i]):
            blk = next_block
            next_block += 1
            block_table[i, j] = blk
            t0, t1 = j * block_size, (j + 1) * block_size
            k_cache[blk] = k_bytes[t0:t1].permute(1, 0, 2)
            v_cache[blk] = v_bytes[t0:t1].permute(1, 0, 2)
            k_scale_cache[blk] = ks_bytes[t0:t1].permute(1, 0, 2, 3)
            r0, r1 = j * rows_per_block, (j + 1) * rows_per_block
            v_scale_cache[blk] = vs_rows[r0:r1].permute(1, 0, 2, 3)

    q_scale_packed = torch.cat(q_scale_all)  # (B, nq, dg//2, 2)
    if q_descale_layout == "N2TGD":
        q_scale_packed = q_scale_tnd_to_n2tgd(q_scale_packed, nkv)

    kwargs = {
        "q": torch.cat(q_all).npu(),
        "k": k_cache.npu().view(FP8_DTYPE),
        "v": v_cache.npu().view(FP8_DTYPE),
        "q_descale": e8m0_npu(q_scale_packed),
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
        "layout_q_descale": q_descale_layout,
        "layout_kv": "PA_BNBD",
        "layout_out": "TND",
        "attn_mask": causal_mask_npu(),
    }
    out = call_qfa(kwargs, nq, nkv, d)
    torch.npu.synchronize()
    print(f"  attn_out: shape={tuple(out.shape)} dtype={out.dtype}")
    return compare(name, out, torch.cat(goldens))


CASES = {
    "TND-SMALL": lambda: run_tnd_case("TND-SMALL", [128, 200], 8, 2, 128, seed=1024),
    "TND-27B": lambda: run_tnd_case("TND-27B", [512, 300], 24, 4, 256, seed=2027),
    "PA-TND": lambda: run_pa_bnbd_case(
        "PA-TND", [300, 257], 8, 2, 128, 128, "TND", seed=3072),
    "PA-N2TGD": lambda: run_pa_bnbd_case(
        "PA-N2TGD", [1, 127, 128, 129, 300], 24, 4, 256, 128, "N2TGD", seed=4096),
}


def main() -> int:
    import torch_npu  # noqa: F401

    torch.npu.set_device(int(os.environ.get("QFA_DEVICE", "0")))
    bootstrap_ops()

    selected = os.environ.get("QFA_CASES")
    names = [n.strip() for n in selected.split(",")] if selected else list(CASES)
    results = {}
    for name in names:
        if name not in CASES:
            print(f"[RED] unknown case {name}; known: {list(CASES)}")
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
        print(f"  {name}: {'GREEN' if ok else 'RED'}")
        all_green &= ok
    print(f"[{'GREEN' if all_green else 'RED'}] quant_flash_attn (official master) "
          f"single-op smoke")
    return 0 if all_green else 1


if __name__ == "__main__":
    sys.exit(main())
