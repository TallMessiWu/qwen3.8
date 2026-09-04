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

Stage-2 M0 cases (integration risk burn-down before touching the engine):

  PA-BBND      the natural vllm-ascend cache shape (block, token, head, dim)
               that stage 2 will feed straight from the paged KV cache
  PA-BBND-MTP  MTP verify form: variable q lengths [4, 2, 1, 3] (mixed accept)
               with N2TGD q_descale on the 27B head shape
  QUANT-FEED   torch_npu.npu_dynamic_mx_quant products fed directly to QFA;
               the golden also consumes the device quantization outputs, so
               green == online-quant and QFA agree on scale semantics
  GRAPH        aclgraph capture/replay in the planned D4 mode (metadata runs
               OUTSIDE the graph into a fixed buffer; main op replays):
               A fixed capture-constant max_seqlen_kv == tight bit-exact,
               B same-input replay == eager bit-exact,
               C in-place buffer swap to a second batch + replay == eager

Every case calls npu_quant_flash_attn_metadata first and the main op second
with an identical argument set (the op does no cross-call validation).
Prints [GREEN]/[RED] per case and exits non-zero on any RED.

Usage (inside the serving container, after pip_install_qfa.sh):
  python scripts/bench/test_qfa_op.py
  QFA_CASES=PA-N2TGD python scripts/bench/test_qfa_op.py   # one case
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


def e8m0_bytes_to_fp32(scale_bytes: torch.Tensor) -> torch.Tensor:
    """uint8 biased exponents -> fp32 2^(b-127); 0xFF (NaN sentinel) -> 0."""
    b = scale_bytes.to(torch.float32)
    out = torch.pow(2.0, b - 127.0)
    return torch.where(scale_bytes == 0xFF, torch.zeros_like(out), out)


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


def _pack_pa_data(q_lens, kv_lens, nq, nkv, d, block_size, kv_layout,
                  total_blocks=None, table_cols=None):
    """Quantize + pack a batch into paged caches (CPU uint8 tensors).

    kv_layout: "PA_BNBD" (block, head, token, dim) or "PA_BBND"
    (block, token, head, dim -- the natural vllm-ascend cache shape).
    total_blocks/table_cols allow a fixed-capacity cache (graph buffers).
    Returns a dict of CPU tensors plus per-sequence goldens.
    """
    b = len(kv_lens)
    softmax_scale = 1.0 / math.sqrt(d)
    dg = ((d + 63) // 64 * 64) // GROUP  # quant groups per token
    rows_per_block = block_size // 64

    blocks_per_seq = [max(1, (s + block_size - 1) // block_size) for s in kv_lens]
    if total_blocks is None:
        total_blocks = sum(blocks_per_seq)
    assert sum(blocks_per_seq) <= total_blocks, "cache capacity too small"
    if table_cols is None:
        table_cols = max(blocks_per_seq)

    if kv_layout == "PA_BNBD":
        kv_shape = (total_blocks, nkv, block_size, d)
        ks_shape = (total_blocks, nkv, block_size, dg // 2, 2)
        vs_shape = (total_blocks, nkv, rows_per_block, d, 2)
    else:  # PA_BBND
        kv_shape = (total_blocks, block_size, nkv, d)
        ks_shape = (total_blocks, block_size, nkv, dg // 2, 2)
        vs_shape = (total_blocks, rows_per_block, nkv, d, 2)
    k_cache = torch.zeros(kv_shape, dtype=torch.uint8)
    v_cache = torch.zeros(kv_shape, dtype=torch.uint8)
    k_scale_cache = torch.zeros(ks_shape, dtype=torch.uint8)
    v_scale_cache = torch.zeros(vs_shape, dtype=torch.uint8)
    block_table = torch.zeros(b, table_cols, dtype=torch.int32)

    q_all, q_scale_all, goldens = [], [], []
    next_block = 0
    for i, (ql, s) in enumerate(zip(q_lens, kv_lens)):
        q = torch.randn(ql, nq, d, dtype=torch.bfloat16)
        if s == 0:
            qs0 = QuantSeq(q, torch.zeros(1, nkv, d, dtype=torch.bfloat16),
                           torch.zeros(1, nkv, d, dtype=torch.bfloat16))
            q_all.append(qs0.q_fp8)
            q_scale_all.append(qs0.qk_scale_packed_tnd("q"))
            goldens.append(torch.zeros(ql, nq, d))
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
        for j in range(blocks_per_seq[i]):
            blk = next_block
            next_block += 1
            block_table[i, j] = blk
            t0, t1 = j * block_size, (j + 1) * block_size
            r0, r1 = j * rows_per_block, (j + 1) * rows_per_block
            if kv_layout == "PA_BNBD":
                k_cache[blk] = k_bytes[t0:t1].permute(1, 0, 2)
                v_cache[blk] = v_bytes[t0:t1].permute(1, 0, 2)
                k_scale_cache[blk] = ks_bytes[t0:t1].permute(1, 0, 2, 3)
                v_scale_cache[blk] = vs_rows[r0:r1].permute(1, 0, 2, 3)
            else:  # PA_BBND: token-major, no transpose
                k_cache[blk] = k_bytes[t0:t1]
                v_cache[blk] = v_bytes[t0:t1]
                k_scale_cache[blk] = ks_bytes[t0:t1]
                v_scale_cache[blk] = vs_rows[r0:r1]

    return {
        "q": torch.cat(q_all),
        "q_scale_packed_tnd": torch.cat(q_scale_all),
        "k_cache": k_cache,
        "v_cache": v_cache,
        "k_scale_cache": k_scale_cache,
        "v_scale_cache": v_scale_cache,
        "block_table": block_table,
        "cu_seqlens_q": torch.tensor(
            [0] + torch.tensor(q_lens).cumsum(0).tolist(), dtype=torch.int32),
        "seqused_kv": torch.tensor(kv_lens, dtype=torch.int32),
        "softmax_scale": softmax_scale,
        "goldens": goldens,
    }


def run_pa_case(name, q_lens, kv_lens, nq, nkv, d, block_size,
                q_descale_layout, kv_layout, seed) -> bool:
    print(f"== {name}: {kv_layout} q={q_lens} B={len(kv_lens)} kv={kv_lens} "
          f"Nq={nq} Nkv={nkv} D={d} block={block_size} "
          f"q_descale={q_descale_layout} ==")
    torch.manual_seed(seed)
    data = _pack_pa_data(q_lens, kv_lens, nq, nkv, d, block_size, kv_layout)

    q_scale_packed = data["q_scale_packed_tnd"]
    if q_descale_layout == "N2TGD":
        q_scale_packed = q_scale_tnd_to_n2tgd(q_scale_packed, nkv)

    kwargs = {
        "q": data["q"].npu(),
        "k": data["k_cache"].npu().view(FP8_DTYPE),
        "v": data["v_cache"].npu().view(FP8_DTYPE),
        "q_descale": e8m0_npu(q_scale_packed),
        "k_descale": e8m0_npu(data["k_scale_cache"]),
        "v_descale": e8m0_npu(data["v_scale_cache"]),
        "block_table": data["block_table"].npu(),
        "cu_seqlens_q": data["cu_seqlens_q"].npu(),
        "seqused_kv": data["seqused_kv"].npu(),
        "softmax_scale": data["softmax_scale"],
        "mask_mode": 3,
        "max_seqlen_q": max(q_lens),
        "max_seqlen_kv": max(kv_lens),
        "layout_q": "TND",
        "layout_q_descale": q_descale_layout,
        "layout_kv": kv_layout,
        "layout_out": "TND",
        "attn_mask": causal_mask_npu(),
    }
    out = call_qfa(kwargs, nq, nkv, d)
    torch.npu.synchronize()
    print(f"  attn_out: shape={tuple(out.shape)} dtype={out.dtype}")
    return compare(name, out, torch.cat(data["goldens"]))


# --------------------------------------------------------------------------
# M0 case: online-quant closed loop (npu_dynamic_mx_quant -> QFA)
# --------------------------------------------------------------------------
def npu_quant_qk_online(x_npu: torch.Tensor):
    """(S, N, D) bf16 on NPU -> fp8 (S, N, D), packed scale bytes
    (S, N, D//64, 2) on CPU, raw scale bytes (S, N, D//32) on CPU."""
    import torch_npu

    s, n, d = x_npu.shape
    fp8, scale = torch_npu.npu_dynamic_mx_quant(
        x_npu.reshape(s * n, d), dst_type=FP8_DTYPE, scale_alg=0)
    raw = scale.view(torch.uint8).reshape(s, n, d // GROUP).cpu()
    packed = raw.reshape(s, n, d // 64, 2)
    return fp8.reshape(s, n, d), packed, raw


def npu_quant_v_online(v_npu: torch.Tensor):
    """(S, N, D) bf16 on NPU, quantized in 32-token groups along S.
    Returns fp8 (S, N, D), packed rows (ceil64(S)//64, N, D, 2) CPU,
    raw scale bytes (N, D, ceil64(S)//32) CPU."""
    import torch_npu

    s, n, d = v_npu.shape
    s_pad = (s + 63) // 64 * 64
    padded = v_npu
    if s_pad != s:
        padded = torch.nn.functional.pad(v_npu, (0, 0, 0, 0, 0, s_pad - s))
    cols = padded.permute(1, 2, 0).contiguous().reshape(n * d, s_pad)
    fp8, scale = torch_npu.npu_dynamic_mx_quant(
        cols, dst_type=FP8_DTYPE, scale_alg=0)
    # fp8 tensors must move through uint8 views (fp8 transpose falls to AICPU)
    v_fp8 = (fp8.view(torch.uint8).reshape(n, d, s_pad)
             .permute(2, 0, 1).contiguous()[:s].view(FP8_DTYPE))
    raw = scale.view(torch.uint8).reshape(n, d, s_pad // GROUP).cpu()
    packed = (raw.reshape(n, d, s_pad // 64, 2)
              .permute(2, 0, 1, 3).contiguous())
    return v_fp8, packed, raw


def run_quant_feed_case(name, seq_lens, nq, nkv, d, seed) -> bool:
    """bf16 -> npu_dynamic_mx_quant on device -> pack -> QFA; the CPU golden
    consumes the DEVICE quantization products, so a green run proves the
    online-quant path and QFA agree on scale semantics end to end."""
    print(f"== {name}: online-quant closed loop TND B={len(seq_lens)} "
          f"S={seq_lens} Nq={nq} Nkv={nkv} D={d} ==")
    assert d % 64 == 0, "online-quant helper assumes 64-aligned D"
    torch.manual_seed(seed)
    softmax_scale = 1.0 / math.sqrt(d)

    per_seq = []
    ref_match_num = 0
    ref_match_den = 0
    for s in seq_lens:
        q_bf = torch.randn(s, nq, d, dtype=torch.bfloat16)
        k_bf = torch.randn(s, nkv, d, dtype=torch.bfloat16)
        v_bf = torch.randn(s, nkv, d, dtype=torch.bfloat16)
        q_fp8, q_packed, q_raw = npu_quant_qk_online(q_bf.npu())
        k_fp8, k_packed, k_raw = npu_quant_qk_online(k_bf.npu())
        v_fp8, v_packed, v_raw = npu_quant_v_online(v_bf.npu())

        # informational: device scales vs the CPU reference formula
        for bf, raw, kind in ((q_bf, q_raw, "qk"), (k_bf, k_raw, "qk")):
            ref = qk_group_scale_bnsd(bf.float().permute(1, 0, 2).unsqueeze(0))
            ref_bytes = fp32_to_e8m0_bytes(
                ref.squeeze(0).permute(1, 0, 2), "ref")  # (S, N, Dg)
            ref_match_num += int((ref_bytes == raw).sum())
            ref_match_den += ref_bytes.numel()

        def fp8_to_cpu(t):
            # fp8 payloads travel as byte views (fp8 copies can hit AICPU)
            return t.view(torch.uint8).cpu().view(FP8_DTYPE)

        sg_real = (s + GROUP - 1) // GROUP
        golden = cpu_golden_one_seq(
            fp8_to_cpu(q_fp8), fp8_to_cpu(k_fp8), fp8_to_cpu(v_fp8),
            e8m0_bytes_to_fp32(q_raw).permute(1, 0, 2).unsqueeze(0),
            e8m0_bytes_to_fp32(k_raw).permute(1, 0, 2).unsqueeze(0),
            e8m0_bytes_to_fp32(v_raw).permute(0, 2, 1)[:, :sg_real].unsqueeze(0),
            softmax_scale, mask_mode=3,
        )
        per_seq.append((q_fp8, k_fp8, v_fp8, q_packed, k_packed, v_packed, golden))

    if ref_match_den:
        print(f"  device-vs-CPU-reference scale byte match: "
              f"{ref_match_num / ref_match_den:.6f} (informational)")

    def cat_fp8(idx):
        return torch.cat([p[idx].view(torch.uint8) for p in per_seq]).view(FP8_DTYPE)

    cu = torch.tensor([0] + torch.tensor(seq_lens).cumsum(0).tolist(), dtype=torch.int32)
    kwargs = {
        "q": cat_fp8(0), "k": cat_fp8(1), "v": cat_fp8(2),
        "q_descale": e8m0_npu(torch.cat([p[3] for p in per_seq])),
        "k_descale": e8m0_npu(torch.cat([p[4] for p in per_seq])),
        "v_descale": e8m0_npu(torch.cat([p[5] for p in per_seq])),
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
    return compare(name, out, torch.cat([p[6] for p in per_seq]))


# --------------------------------------------------------------------------
# M0 case: aclgraph capture/replay smoke (D4 mode: metadata outside the graph)
# --------------------------------------------------------------------------
def run_graph_case(name, kv_lens_a, kv_lens_b, nq, nkv, d, block_size, seed) -> bool:
    """Three checks on PA_BBND decode q=1:
      A  eager: tight max_seqlen_kv vs a fixed capture constant -> bit-exact
      B  capture main op (metadata precomputed into a fixed buffer), replay
         with unchanged inputs -> bit-exact vs eager
      C  overwrite every input buffer in place with a second batch, rerun
         metadata outside the graph, replay -> bit-exact vs eager on batch 2
    """
    b = len(kv_lens_a)
    assert len(kv_lens_b) == b
    max_kv_const = 2048
    table_cols = max_kv_const // block_size
    cap_blocks = b * table_cols
    q_lens = [1] * b
    print(f"== {name}: aclgraph PA_BBND decode q=1 B={b} kv_a={kv_lens_a} "
          f"kv_b={kv_lens_b} Nq={nq} Nkv={nkv} D={d} block={block_size} "
          f"max_kv_const={max_kv_const} ==")
    torch.manual_seed(seed)
    data_a = _pack_pa_data(q_lens, kv_lens_a, nq, nkv, d, block_size, "PA_BBND",
                           total_blocks=cap_blocks, table_cols=table_cols)
    data_b = _pack_pa_data(q_lens, kv_lens_b, nq, nkv, d, block_size, "PA_BBND",
                           total_blocks=cap_blocks, table_cols=table_cols)
    for data in (data_a, data_b):
        data["q_n2tgd"] = q_scale_tnd_to_n2tgd(data["q_scale_packed_tnd"], nkv)

    # fixed-address device buffers (uint8 for all quantized payloads)
    buf_names = ("k_cache", "v_cache", "k_scale_cache", "v_scale_cache",
                 "block_table", "seqused_kv")
    bufs = {key: data_a[key].npu() for key in buf_names}
    q_buf = data_a["q"].view(torch.uint8).npu()
    qd_buf = data_a["q_n2tgd"].npu()
    cu_q = data_a["cu_seqlens_q"].npu()
    metadata_buf = torch.zeros(4096, dtype=torch.int32).npu()
    mask = causal_mask_npu()
    softmax_scale = data_a["softmax_scale"]

    def common(max_kv):
        return {
            "cu_seqlens_q": cu_q,
            "seqused_kv": bufs["seqused_kv"],
            "mask_mode": 3,
            "max_seqlen_q": 1,
            "max_seqlen_kv": max_kv,
            "layout_q": "TND",
            "layout_q_descale": "N2TGD",
            "layout_kv": "PA_BBND",
            "layout_out": "TND",
        }

    def refresh_metadata(max_kv):
        md = torch.ops._C_ascend.npu_quant_flash_attn_metadata(
            nq, nkv, d, 1,
            v_descale=bufs["v_scale_cache"].view(torch.float8_e8m0fnu),
            **common(max_kv),
        )
        metadata_buf.copy_(md)

    def run_main(max_kv, metadata):
        out, _ = torch.ops._C_ascend.npu_quant_flash_attn(
            q_buf.view(FP8_DTYPE),
            bufs["k_cache"].view(FP8_DTYPE),
            bufs["v_cache"].view(FP8_DTYPE),
            qd_buf.view(torch.float8_e8m0fnu),
            bufs["k_scale_cache"].view(torch.float8_e8m0fnu),
            bufs["v_scale_cache"].view(torch.float8_e8m0fnu),
            1,
            block_table=bufs["block_table"],
            attn_mask=mask,
            metadata=metadata,
            softmax_scale=softmax_scale,
            **common(max_kv),
        )
        return out

    def load_batch(data):
        for key in buf_names:
            bufs[key].copy_(data[key])
        q_buf.copy_(data["q"].view(torch.uint8))
        qd_buf.copy_(data["q_n2tgd"])

    ok = True

    # -- A: fixed capture-constant max_seqlen_kv must not change the result --
    tight = max(kv_lens_a)
    refresh_metadata(tight)
    ref_tight = run_main(tight, metadata_buf).cpu()
    refresh_metadata(max_kv_const)
    ref_const = run_main(max_kv_const, metadata_buf).cpu()
    torch.npu.synchronize()
    exact_a = torch.equal(ref_tight, ref_const)
    print(f"  [A] tight({tight}) vs const({max_kv_const}) bit-exact={exact_a}")
    ok &= exact_a
    ok &= compare(f"{name}-eager", ref_const, torch.cat(data_a["goldens"]))

    # -- B: capture with metadata outside the graph, replay unchanged --
    for _ in range(2):  # warmup
        refresh_metadata(max_kv_const)
        run_main(max_kv_const, metadata_buf)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out_buf = run_main(max_kv_const, metadata_buf)
    graph.replay()
    torch.npu.synchronize()
    exact_b = torch.equal(out_buf.cpu(), ref_const)
    print(f"  [B] replay(same inputs) vs eager bit-exact={exact_b}")
    ok &= exact_b

    # -- C: swap every input in place to batch 2, metadata outside, replay --
    load_batch(data_b)
    refresh_metadata(max_kv_const)
    ref_b = run_main(max_kv_const, metadata_buf).cpu()
    graph.replay()
    torch.npu.synchronize()
    exact_c = torch.equal(out_buf.cpu(), ref_b)
    print(f"  [C] replay(batch 2) vs eager bit-exact={exact_c}")
    ok &= exact_c
    ok &= compare(f"{name}-replay", out_buf, torch.cat(data_b["goldens"]))

    print(f"  [{name}] {'GREEN' if ok else 'RED'}")
    return ok


CASES = {
    "TND-SMALL": lambda: run_tnd_case("TND-SMALL", [128, 200], 8, 2, 128, seed=1024),
    "TND-27B": lambda: run_tnd_case("TND-27B", [512, 300], 24, 4, 256, seed=2027),
    "PA-TND": lambda: run_pa_case(
        "PA-TND", [1, 1], [300, 257], 8, 2, 128, 128, "TND", "PA_BNBD", seed=3072),
    "PA-N2TGD": lambda: run_pa_case(
        "PA-N2TGD", [1] * 5, [1, 127, 128, 129, 300], 24, 4, 256, 128,
        "N2TGD", "PA_BNBD", seed=4096),
    # -- M0 additions (stage-2 integration risk burn-down) --
    "PA-BBND": lambda: run_pa_case(
        "PA-BBND", [1, 1], [300, 257], 8, 2, 128, 128, "TND", "PA_BBND", seed=5120),
    "PA-BBND-MTP": lambda: run_pa_case(
        "PA-BBND-MTP", [4, 2, 1, 3], [130, 127, 64, 300], 24, 4, 256, 128,
        "N2TGD", "PA_BBND", seed=6144),
    "QUANT-FEED": lambda: run_quant_feed_case(
        "QUANT-FEED", [192, 130], 24, 4, 256, seed=7168),
    "GRAPH": lambda: run_graph_case(
        "GRAPH", [300, 257, 128, 65], [513, 100, 300, 1], 24, 4, 256, 128,
        seed=8192),
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
