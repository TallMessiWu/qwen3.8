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
  case 5 C-SHAPES milestone-C decode/verify/chunked shapes on PA_BBND @
                  block_size=128 (officially zero-covered): 5a decode q=1
                  (mask 0 vs 3 equivalence, kv boundaries), 5b MTP verify
                  q=4 + mixed accept lengths, 5c chunked-prefill mixed batch
                  with PREFILL(TND descale) x PA_BBND, 5e seqused=0 and 0x00
                  scale-slot corners, 5f a fixed max_seqlen_kv (what a captured
                  aclgraph must carry) vs a tight one.
                  QFA_CASES=C-SHAPES runs just this case.
  case 6 COMPOSE  writes a KV cache with the milestone-C impl's own algorithm,
                  byte-compares it against the golden packing, then reads it
                  back through QFA - the composition that W1/W2 and case 5
                  each only half-cover. QFA_CASES=COMPOSE runs just this case.
  case 7 BATCH-STEPS  prefills a 3-request batch then steps decode with the
                  impl's own write algorithms, reading the cache back after
                  every step and naming the first step/request that diverges.
                  QFA_STEPS sets the step count (default 6).
  case 8 SCHEDULE replays the step pattern a live engine actually produces:
                  the first prompt prefills alone, then a mixed batch decodes
                  it while the others prefill, then pure decode steps. This is
                  the shape the end-to-end smoke takes and the earlier cases
                  never do. QFA_CASES=SCHEDULE runs just this case.
  case 9 SHAPES   runs that same schedule over four block-table geometries,
                  stepping from the hand-built one to the engine's real one
                  (36 columns, ids far from zero, thousands of blocks). On
                  hardware the live decode read is correct for batch row 0 and
                  wrong for rows 1 and 2, which no earlier case reproduces.
                  QFA_CASES=SHAPES runs just this case.

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


# --------------------------------------------------------------------------
# Case 5: milestone-C decode/verify/chunked shapes on PA_BBND @ block_size=128
# (officially zero-covered combination; 27B per-rank Nq=24 Nkv=4 D=256)
# --------------------------------------------------------------------------
def _run_pa_bbnd(
    name, q_lens, kv_lens, nq, nkv, d, bs, mask_mode, q_descale_layout, max_kv=None, ret_out=False, seed=None
):
    """Generic PA_BBND runner: CPU-packed caches, per-seq golden, one QFA call."""
    torch.manual_seed(seed if seed is not None else sum(map(ord, name)))  # deterministic across runs
    b = len(kv_lens)
    g = nq // nkv
    softmax_scale = 1.0 / math.sqrt(d)

    blocks_per_seq = [max(1, (s + bs - 1) // bs) for s in kv_lens]
    total_blocks = sum(blocks_per_seq)
    k_cache = torch.zeros(total_blocks, bs, nkv, d, dtype=torch.uint8)
    v_cache = torch.zeros(total_blocks, bs, nkv, d, dtype=torch.uint8)
    k_scale_cache = torch.zeros(total_blocks, bs, nkv, d // 64, 2, dtype=torch.uint8)
    # 0x00 == 2^-127: mirrors milestone-C zeros-initialized scale planes (5e)
    v_scale_cache = torch.zeros(total_blocks, bs // 64, nkv, d, 2, dtype=torch.uint8)
    block_table = torch.zeros(b, max(blocks_per_seq), dtype=torch.int32)

    q_all, qs_all, goldens = [], [], []
    next_block = 0
    for i, (ql, s) in enumerate(zip(q_lens, kv_lens)):
        q = torch.randn(max(ql, 1), nq, d, dtype=torch.bfloat16)
        qsc = qk_group_scale(q)
        q_fp8 = quantize_with_scale(q, qsc.repeat_interleave(GROUP, dim=-1)[..., :d])
        q_all.append(q_fp8[:ql])
        qs_all.append(pack_last_pairs(fp32_to_e8m0_bytes(qsc, "q_scale"))[:ql])

        if s == 0:
            goldens.append(torch.zeros(ql, nq, d))
            block_table[i, 0] = next_block
            next_block += 1
            continue
        k = torch.randn(s, nkv, d, dtype=torch.bfloat16)
        v = torch.randn(s, nkv, d, dtype=torch.bfloat16)
        ksc = qk_group_scale(k)
        s_pad = blocks_per_seq[i] * bs
        v_pad = torch.nn.functional.pad(v.float(), (0, 0, 0, 0, 0, s_pad - s))
        vsc = v_group_scale(v_pad)
        k_fp8 = quantize_with_scale(k, ksc.repeat_interleave(GROUP, dim=-1)[..., :d])
        v_fp8 = quantize_with_scale(v_pad, vsc.repeat_interleave(GROUP, dim=0)[:s_pad])[:s]
        goldens.append(
            cpu_golden_one_seq(
                q_fp8[:ql], k_fp8, v_fp8, qsc[:ql], ksc,
                vsc[: (s + GROUP - 1) // GROUP], softmax_scale, mask_mode == 3,
            )
        )
        k_bytes = torch.nn.functional.pad(k_fp8.view(torch.uint8), (0, 0, 0, 0, 0, s_pad - s))
        v_bytes = torch.nn.functional.pad(v_fp8.view(torch.uint8), (0, 0, 0, 0, 0, s_pad - s))
        ks_bytes = pack_last_pairs(fp32_to_e8m0_bytes(ksc, "k_scale"))
        ks_bytes = torch.nn.functional.pad(ks_bytes, (0, 0, 0, 0, 0, 0, 0, s_pad - s))
        vs_bytes = pack_v_scale_seq(fp32_to_e8m0_bytes(vsc, "v_scale"))
        for j in range(blocks_per_seq[i]):
            blk = next_block
            next_block += 1
            block_table[i, j] = blk
            k_cache[blk] = k_bytes[j * bs : (j + 1) * bs]
            v_cache[blk] = v_bytes[j * bs : (j + 1) * bs]
            k_scale_cache[blk] = ks_bytes[j * bs : (j + 1) * bs]
            v_scale_cache[blk] = vs_bytes[j * (bs // 64) : (j + 1) * (bs // 64)]

    qs = torch.cat(qs_all)  # (Tq, nq, d//64, 2)
    if q_descale_layout == "N2TGD":
        tq = qs.shape[0]
        qs = qs.reshape(tq, nkv, g, d // 64, 2).permute(1, 0, 2, 3, 4).contiguous()
    cu_q = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32)
    npu_kwargs = {
        "q": torch.cat(q_all).npu(),
        "k": k_cache.npu().view(torch.float8_e4m3fn),
        "v": v_cache.npu().view(torch.float8_e4m3fn),
        "q_descale": e8m0_npu(qs),
        "k_descale": e8m0_npu(k_scale_cache),
        "v_descale": e8m0_npu(v_scale_cache),
        "block_table": block_table.npu(),
        "cu_seqlens_q": cu_q.npu(),
        "seqused_kv": torch.tensor(kv_lens, dtype=torch.int32).npu(),
        "softmax_scale": softmax_scale,
        "mask_mode": mask_mode,
        "max_seqlen_q": max(q_lens),
        "max_seqlen_kv": max_kv or max(kv_lens),
        "layout_q": "TND",
        "layout_q_descale": q_descale_layout,
        "layout_kv": "PA_BBND",
        "layout_out": "TND",
    }
    mask = causal_mask_npu() if mask_mode == 3 else None
    out = call_qfa(npu_kwargs, nq, nkv, d, mask)
    if ret_out:
        return out
    print(f"  attn_out: shape={tuple(out.shape)} dtype={out.dtype}")
    return compare(name, out, torch.cat(goldens))


def case_c_decode_shapes() -> bool:
    print("== case 5: milestone-C shapes on PA_BBND@128 (27B: Nq=24 Nkv=4 D=256) ==")
    nq, nkv, d, bs = 24, 4, 256, 128
    ok = True
    print("-- 5a decode q=1, mask_mode 0 vs 3 equivalence, boundary kv --")
    kv = [1, 127, 128, 129, 300]
    ok &= _run_pa_bbnd("5a-mask0", [1] * 5, kv, nq, nkv, d, bs, 0, "N2TGD")
    ok &= _run_pa_bbnd("5a-mask3", [1] * 5, kv, nq, nkv, d, bs, 3, "N2TGD")
    print("-- 5b MTP verify q=4 (uniform) + mixed accept lengths --")
    ok &= _run_pa_bbnd("5b-q4", [4, 4, 4], [68, 130, 257], nq, nkv, d, bs, 3, "N2TGD")
    ok &= _run_pa_bbnd("5b-var", [4, 2, 1, 3], [66, 130, 200, 41], nq, nkv, d, bs, 3, "N2TGD")
    print("-- 5c chunked-prefill mixed batch, PREFILL(TND descale) x PA_BBND --")
    ok &= _run_pa_bbnd("5c-mix", [1, 512], [300, 1536], nq, nkv, d, bs, 3, "TND")
    print("-- 5e zero/padding corners (seqused=0 req, 0x00 scale slots) --")
    ok &= _run_pa_bbnd("5e-zero", [1, 1], [0, 65], nq, nkv, d, bs, 3, "N2TGD")
    print("-- 5f max_seqlen_kv as a capture constant vs a tight bound --")
    args = ([4, 4], [130, 257], nq, nkv, d, bs, 3, "N2TGD")
    tight = _run_pa_bbnd("5f-tight", *args, ret_out=True, seed=5150)
    loose = _run_pa_bbnd("5f-const", *args, max_kv=8192, ret_out=True, seed=5150)
    same = torch.equal(tight.cpu(), loose.cpu())
    print(f"  [5f] tight(257) vs constant(8192) bit-exact={same}")
    print(f"  [5f] {'GREEN' if same else 'RED'} (C3 capture needs a fixed max_seqlen_kv)")
    ok &= same
    print(f"  [C-SHAPES] {'GREEN' if ok else 'RED'}")
    return ok



# --------------------------------------------------------------------------
# Case 6: write-then-read composition (the milestone-C impl's own algorithm)
# --------------------------------------------------------------------------
def _impl_quant_key(key):
    """Port of AscendAttentionBackendImpl._qfa_write_key's quant step."""
    import torch_npu

    t, n, d = key.shape
    fp8, scale = torch_npu.npu_dynamic_mx_quant(key.reshape(t * n, d), dst_type=FP8, scale_alg=0)
    return (
        fp8.reshape(t, n, d).view(torch.uint8),
        scale.view(torch.uint8).reshape(t, n, d // 64, 2),
    )


def _impl_quant_windows(rows):
    """Port of _qfa_quant_along_tokens: (W,64,N,D) bf16 -> bytes + (W,N,D,2)."""
    import torch_npu

    w, group, n, d = rows.shape
    cols = rows.permute(0, 2, 3, 1).reshape(w * n * d, group)
    fp8, scale = torch_npu.npu_dynamic_mx_quant(cols.contiguous(), dst_type=FP8, scale_alg=0)
    fp8 = fp8.view(torch.uint8).reshape(w, n, d, group).permute(0, 3, 1, 2)
    return fp8.contiguous(), scale.view(torch.uint8).reshape(w, n, d, 2)


def case_write_read_composition() -> bool:
    """Does a cache written by the impl's algorithm read back correctly?

    W1/W2 check the write against a CPU reference and case 5 checks the read
    against caches packed by the golden; neither covers the composition, so a
    packing mismatch between the two halves would leave both green and still
    serve garbage. This writes with the impl's algorithm, byte-compares the
    result against the golden packing, then reads it back through QFA.
    """
    print("== case 6: impl write -> QFA read (27B shape, multi-request batch) ==")
    torch.manual_seed(606)
    nq, nkv, d, bs = 24, 4, 256, 128
    g = nq // nkv
    prompt_lens = [5, 10, 9]  # mirrors the smoke prompts: one short, two longer
    b = len(prompt_lens)
    softmax_scale = 1.0 / math.sqrt(d)
    blocks_per_req = 2
    total_blocks = b * blocks_per_req
    block_table = torch.arange(total_blocks, dtype=torch.int32).reshape(b, blocks_per_req)

    k_fp8 = torch.zeros(total_blocks, bs, nkv, d, dtype=torch.uint8).npu()
    k_scale = torch.zeros(total_blocks, bs, nkv, d // 64, 2, dtype=torch.uint8).npu()
    v_fp8 = torch.zeros(total_blocks, bs, nkv, d, dtype=torch.uint8).npu()
    v_scale = torch.zeros(total_blocks, bs // 64, nkv, d, 2, dtype=torch.uint8).npu()

    keys = [torch.randn(s, nkv, d, dtype=torch.bfloat16) for s in prompt_lens]
    values = [torch.randn(s, nkv, d, dtype=torch.bfloat16) for s in prompt_lens]

    # ---- impl write path (bulk / prefill) ----
    key_cat = torch.cat(keys).npu()
    slots = torch.cat(
        [int(block_table[i, 0]) * bs + torch.arange(s) for i, s in enumerate(prompt_lens)]
    ).npu()
    kf, ks = _impl_quant_key(key_cat)
    k_fp8.view(-1, nkv, d).index_put_((slots.long(),), kf)
    k_scale.view(-1, nkv, d // 64, 2).index_put_((slots.long(),), ks)

    bt_npu = block_table.npu()
    for i, s in enumerate(prompt_lens):
        num_windows = (s - 1) // 64 + 1
        buf = torch.zeros(num_windows * 64, nkv, d, dtype=torch.bfloat16).npu()
        buf[:s] = values[i].npu()
        vf, vsc = _impl_quant_windows(buf.view(num_windows, 64, nkv, d))
        windows = torch.arange(num_windows).npu()
        blocks = bt_npu[i, torch.div(windows, 2, rounding_mode="floor")].long()
        window_slots = blocks * 2 + (windows % 2)
        v_fp8.view(-1, 64, nkv, d).index_put_((window_slots,), vf)
        v_scale.view(-1, nkv, d, 2).index_put_((window_slots,), vsc)
    torch.npu.synchronize()

    # ---- byte-compare against the golden packing ----
    byte_ok = True
    for i, s in enumerate(prompt_lens):
        blk = int(block_table[i, 0])
        ref_ksc = qk_group_scale(keys[i])
        ref_k = quantize_with_scale(keys[i], ref_ksc.repeat_interleave(GROUP, dim=-1)[..., :d])
        k_match = torch.equal(k_fp8[blk, :s].cpu(), ref_k.view(torch.uint8))
        ks_match = torch.equal(k_scale[blk, :s].cpu(), pack_last_pairs(fp32_to_e8m0_bytes(ref_ksc, "k")))

        s_pad = (s + 63) // 64 * 64
        v_pad = torch.nn.functional.pad(values[i].float(), (0, 0, 0, 0, 0, s_pad - s))
        ref_vsc = v_group_scale(v_pad)
        ref_v = quantize_with_scale(v_pad, ref_vsc.repeat_interleave(GROUP, dim=0)[:s_pad])[:s]
        v_match = torch.equal(v_fp8[blk, :s].cpu(), ref_v.view(torch.uint8))
        # Groups made entirely of padding have no canonical scale: the golden
        # calls them 2^0 and the NPU quantizer 2^-127, and either way they
        # multiply zeroed V to zero. Compare only groups holding real tokens.
        got_vs = v_scale[blk, : (s_pad // 64)].cpu()
        exp_vs = pack_v_scale_seq(fp32_to_e8m0_bytes(ref_vsc, "v"))
        real_groups = (s + GROUP - 1) // GROUP
        vs_match = True
        for grp in range(real_groups):
            row, half = grp // 2, grp % 2
            vs_match = vs_match and torch.equal(got_vs[row, ..., half], exp_vs[row, ..., half])
        print(
            f"  [6] req {i} (len={s}): k={k_match} k_scale={ks_match} v={v_match} v_scale={vs_match}"
        )
        byte_ok = byte_ok and k_match and ks_match and v_match and vs_match

    # ---- read it back with QFA, one decode token per request ----
    q_all, qs_all, goldens = [], [], []
    for i, s in enumerate(prompt_lens):
        q = torch.randn(1, nq, d, dtype=torch.bfloat16)
        qsc = qk_group_scale(q)
        q_fp8 = quantize_with_scale(q, qsc.repeat_interleave(GROUP, dim=-1)[..., :d])
        q_all.append(q_fp8)
        qs_all.append(pack_last_pairs(fp32_to_e8m0_bytes(qsc, "q")))
        ref_ksc = qk_group_scale(keys[i])
        ref_k = quantize_with_scale(keys[i], ref_ksc.repeat_interleave(GROUP, dim=-1)[..., :d])
        s_pad = (s + 63) // 64 * 64
        v_pad = torch.nn.functional.pad(values[i].float(), (0, 0, 0, 0, 0, s_pad - s))
        ref_vsc = v_group_scale(v_pad)
        ref_v = quantize_with_scale(v_pad, ref_vsc.repeat_interleave(GROUP, dim=0)[:s_pad])[:s]
        goldens.append(
            cpu_golden_one_seq(
                q_fp8, ref_k, ref_v, qsc, ref_ksc,
                ref_vsc[: (s + GROUP - 1) // GROUP], softmax_scale, True,
            )
        )
    qs = torch.cat(qs_all).reshape(b, nkv, g, d // 64, 2).permute(1, 0, 2, 3, 4).contiguous()
    npu_kwargs = {
        "q": torch.cat(q_all).npu(),
        "k": k_fp8.view(torch.float8_e4m3fn),
        "v": v_fp8.view(torch.float8_e4m3fn),
        "q_descale": e8m0_npu(qs),
        "k_descale": k_scale.view(torch.float8_e8m0fnu),
        "v_descale": v_scale.view(torch.float8_e8m0fnu),
        "block_table": block_table.npu(),
        "cu_seqlens_q": torch.arange(b + 1, dtype=torch.int32).npu(),
        "seqused_kv": torch.tensor(prompt_lens, dtype=torch.int32).npu(),
        "softmax_scale": softmax_scale,
        "mask_mode": 3,
        "max_seqlen_q": 1,
        "max_seqlen_kv": max(prompt_lens),
        "layout_q": "TND",
        "layout_q_descale": "N2TGD",
        "layout_kv": "PA_BBND",
        "layout_out": "TND",
    }
    out = call_qfa(npu_kwargs, nq, nkv, d, causal_mask_npu())
    read_ok = compare("6-read", out, torch.cat(goldens))
    ok = byte_ok and read_ok
    print(f"  [COMPOSE] {'GREEN' if ok else 'RED'}")
    return ok



# --------------------------------------------------------------------------
# Case 7: multi-step batched decode (the shape the end-to-end smoke fails on)
# --------------------------------------------------------------------------
def _impl_write_value_decode(staging, v_fp8, v_scale, value, req_ids, positions, ctx_lens, lens, bt, b):
    """Port of AscendAttentionBackendImpl._qfa_write_value_decode."""
    group, n, d = 64, value.shape[1], value.shape[2]
    staging[req_ids, positions % 128] = value

    reqs = torch.arange(b, device=value.device)
    last_window = torch.div(lens - 1, group, rounding_mode="floor").clamp(min=0)
    first_window = torch.minimum(torch.div(ctx_lens, group, rounding_mode="floor"), last_window)
    windows = torch.stack([first_window, last_window], dim=1).reshape(-1)
    win_reqs = reqs.repeat_interleave(2)

    rows = staging.view(-1, 2, group, n, d)[win_reqs, windows % 2]
    valid = (lens.repeat_interleave(2) - windows * group).clamp(0, group)
    keep = torch.arange(group, device=value.device).unsqueeze(0) < valid.unsqueeze(1)
    rows = rows * keep.view(-1, group, 1, 1)

    fp8, scale = _impl_quant_windows(rows)
    blocks = bt[win_reqs, torch.div(windows, 2, rounding_mode="floor")].long()
    slots = blocks * 2 + (windows % 2)
    v_fp8.view(-1, group, n, d).index_put_((slots,), fp8)
    v_scale.view(-1, n, d, 2).index_put_((slots,), scale)


def case_batched_decode_steps() -> bool:
    """Prefill a 3-request batch, then step decode, checking the cache each step.

    The end-to-end smoke is correct for a single request and wrong for the same
    request inside a batch, so the fault lives in the per-request bookkeeping
    across steps rather than in the op or the packing. This drives the impl's
    own write algorithms over several decode steps and reads the cache back
    after each one, naming the first step and request that diverges.
    """
    print("== case 7: batched prefill + multi-step decode (impl algorithms) ==")
    torch.manual_seed(707)
    nq, nkv, d, bs = 24, 4, 256, 128
    g = nq // nkv
    prompt_lens = [5, 10, 9]
    b = len(prompt_lens)
    steps = int(os.environ.get("QFA_STEPS", "6"))
    softmax_scale = 1.0 / math.sqrt(d)
    blocks_per_req = 2
    total_blocks = b * blocks_per_req
    block_table = torch.arange(total_blocks, dtype=torch.int32).reshape(b, blocks_per_req)
    bt_npu = block_table.npu()

    k_fp8 = torch.zeros(total_blocks, bs, nkv, d, dtype=torch.uint8).npu()
    k_scale = torch.zeros(total_blocks, bs, nkv, d // 64, 2, dtype=torch.uint8).npu()
    v_fp8 = torch.zeros(total_blocks, bs, nkv, d, dtype=torch.uint8).npu()
    v_scale = torch.zeros(total_blocks, bs // 64, nkv, d, 2, dtype=torch.uint8).npu()
    staging = torch.zeros(b + 1, 128, nkv, d, dtype=torch.bfloat16).npu()

    # Full history per request, revealed one step at a time.
    total = [s + steps for s in prompt_lens]
    keys = [torch.randn(t, nkv, d, dtype=torch.bfloat16).npu() for t in total]
    values = [torch.randn(t, nkv, d, dtype=torch.bfloat16).npu() for t in total]

    def slots_for(req, lo, hi):
        pos = torch.arange(lo, hi, device="npu")
        return (bt_npu[req, torch.div(pos, bs, rounding_mode="floor")].long() * bs + pos % bs)

    # ---- step 0: prefill all three (bulk path) ----
    key_cat = torch.cat([keys[i][: prompt_lens[i]] for i in range(b)])
    kf, ks = _impl_quant_key(key_cat)
    all_slots = torch.cat([slots_for(i, 0, prompt_lens[i]) for i in range(b)])
    k_fp8.view(-1, nkv, d).index_put_((all_slots,), kf)
    k_scale.view(-1, nkv, d // 64, 2).index_put_((all_slots,), ks)
    for i, s in enumerate(prompt_lens):
        num_windows = (s - 1) // 64 + 1
        buf = torch.zeros(num_windows * 64, nkv, d, dtype=torch.bfloat16).npu()
        buf[:s] = values[i][:s]
        vf, vsc = _impl_quant_windows(buf.view(num_windows, 64, nkv, d))
        windows = torch.arange(num_windows).npu()
        blocks = bt_npu[i, torch.div(windows, 2, rounding_mode="floor")].long()
        v_fp8.view(-1, 64, nkv, d).index_put_((blocks * 2 + windows % 2,), vf)
        v_scale.view(-1, nkv, d, 2).index_put_((blocks * 2 + windows % 2,), vsc)
        staging[i, torch.arange(s).npu() % 128] = buf[:s]
    torch.npu.synchronize()

    cur = list(prompt_lens)
    ok = True
    for step in range(1, steps + 1):
        # ---- write one new token per request (decode path) ----
        new_key = torch.stack([keys[i][cur[i]] for i in range(b)])
        new_value = torch.stack([values[i][cur[i]] for i in range(b)])
        slots = torch.stack([slots_for(i, cur[i], cur[i] + 1)[0] for i in range(b)])
        kf, ks = _impl_quant_key(new_key)
        k_fp8.view(-1, nkv, d).index_put_((slots,), kf)
        k_scale.view(-1, nkv, d // 64, 2).index_put_((slots,), ks)

        ctx_lens = torch.tensor(cur, dtype=torch.int32).npu()
        lens = ctx_lens + 1
        req_ids = torch.arange(b).npu()
        positions = ctx_lens.clone()
        _impl_write_value_decode(
            staging, v_fp8, v_scale, new_value, req_ids, positions, ctx_lens, lens, bt_npu, b
        )
        cur = [c + 1 for c in cur]
        torch.npu.synchronize()

        # ---- read the whole batch back and compare to golden ----
        q_all, qs_all, goldens = [], [], []
        for i in range(b):
            s = cur[i]
            q = torch.randn(1, nq, d, dtype=torch.bfloat16)
            qsc = qk_group_scale(q)
            q_fp8 = quantize_with_scale(q, qsc.repeat_interleave(GROUP, dim=-1)[..., :d])
            q_all.append(q_fp8)
            qs_all.append(pack_last_pairs(fp32_to_e8m0_bytes(qsc, "q")))
            k_cpu = keys[i][:s].cpu()
            v_cpu = values[i][:s].cpu()
            ref_ksc = qk_group_scale(k_cpu)
            ref_k = quantize_with_scale(k_cpu, ref_ksc.repeat_interleave(GROUP, dim=-1)[..., :d])
            s_pad = (s + 63) // 64 * 64
            v_pad = torch.nn.functional.pad(v_cpu.float(), (0, 0, 0, 0, 0, s_pad - s))
            ref_vsc = v_group_scale(v_pad)
            ref_v = quantize_with_scale(v_pad, ref_vsc.repeat_interleave(GROUP, dim=0)[:s_pad])[:s]
            goldens.append(
                cpu_golden_one_seq(
                    q_fp8, ref_k, ref_v, qsc, ref_ksc,
                    ref_vsc[: (s + GROUP - 1) // GROUP], softmax_scale, True,
                )
            )
        qs = torch.cat(qs_all).reshape(b, nkv, g, d // 64, 2).permute(1, 0, 2, 3, 4).contiguous()
        npu_kwargs = {
            "q": torch.cat(q_all).npu(),
            "k": k_fp8.view(torch.float8_e4m3fn),
            "v": v_fp8.view(torch.float8_e4m3fn),
            "q_descale": e8m0_npu(qs),
            "k_descale": k_scale.view(torch.float8_e8m0fnu),
            "v_descale": v_scale.view(torch.float8_e8m0fnu),
            "block_table": bt_npu,
            "cu_seqlens_q": torch.arange(b + 1, dtype=torch.int32).npu(),
            "seqused_kv": torch.tensor(cur, dtype=torch.int32).npu(),
            "softmax_scale": softmax_scale,
            "mask_mode": 3,
            "max_seqlen_q": 1,
            "max_seqlen_kv": max(cur),
            "layout_q": "TND",
            "layout_q_descale": "N2TGD",
            "layout_kv": "PA_BBND",
            "layout_out": "TND",
        }
        out = call_qfa(npu_kwargs, nq, nkv, d, causal_mask_npu()).float().cpu()
        bad = []
        for i in range(b):
            ref = goldens[i].float()
            got = out[i : i + 1]
            cos = torch.nn.functional.cosine_similarity(
                got.reshape(-1), ref.reshape(-1), dim=0
            ).item()
            if cos < 0.999:
                bad.append((i, cur[i], round(cos, 5)))
        status = "ok" if not bad else f"DIVERGED {bad}"
        print(f"  [7] step {step} lens={cur}: {status}")
        if bad:
            ok = False
            break
    print(f"  [BATCH-STEPS] {'GREEN' if ok else 'RED'}")
    return ok



# --------------------------------------------------------------------------
# Case 8: the schedule the runtime actually produces (mixed prefill+decode)
# --------------------------------------------------------------------------
def _impl_write_value_bulk(staging, v_fp8, v_scale, value, cu_q, seq_lens_list, bt, num_reqs):
    """Port of AscendAttentionBackendImpl._qfa_write_value_bulk."""
    group, n, d = 64, value.shape[1], value.shape[2]
    q_start = 0
    for req in range(num_reqs):
        q_end = cu_q[req]
        q_len = q_end - q_start
        new_len = seq_lens_list[req]
        if q_len <= 0 or new_len <= 0:
            q_start = q_end
            continue
        ctx_len = new_len - q_len
        first_window = ctx_len // group
        num_windows = (new_len - 1) // group - first_window + 1
        lead = ctx_len - first_window * group

        buf = value.new_zeros(num_windows * group, n, d)
        if lead > 0:
            ring_rows = torch.arange(first_window * group, ctx_len, device=value.device) % 128
            buf[:lead] = staging[req, ring_rows]
        buf[lead : lead + q_len] = value[q_start:q_end]

        windows = torch.arange(first_window, first_window + num_windows, device=value.device)
        blocks = bt[req, torch.div(windows, 2, rounding_mode="floor")].long()
        fp8, scale = _impl_quant_windows(buf.view(num_windows, group, n, d))
        slots = blocks * 2 + (windows % 2)
        v_fp8.view(-1, group, n, d).index_put_((slots,), fp8)
        v_scale.view(-1, n, d, 2).index_put_((slots,), scale)

        buf_start = first_window * group
        ring_lo = max(new_len - 128, buf_start)
        tail_positions = torch.arange(ring_lo, new_len, device=value.device)
        staging[req, tail_positions % 128] = buf[tail_positions - buf_start]
        q_start = q_end


def _run_schedule(total_blocks: int, bt_cols: int, base_ids: list, label: str) -> bool:
    """Replay the step pattern a live engine produces for a 3-prompt batch.

    The runtime does not prefill every prompt together: it prefills the first
    one alone, then runs a mixed batch where that request decodes while the
    others prefill, and only afterwards settles into pure decode steps. The
    earlier cases all assumed the all-together shape, which is exactly the
    scheduling the end-to-end smoke never takes.

    The block-table geometry is a parameter because the live engine hands the
    op a far wider and sparser table than a hand-built test does, and the
    decode read is wrong there for every batch row but the first.
    """
    print(f"== {label} ==")
    torch.manual_seed(808)
    nq, nkv, d, bs = 24, 4, 256, 128
    g = nq // nkv
    prompt_lens = [5, 5, 9]  # what the smoke actually tokenizes to
    b = len(prompt_lens)
    steps = int(os.environ.get("QFA_STEPS", "4"))
    softmax_scale = 1.0 / math.sqrt(d)
    # Rows hold consecutive 128-token block ids, as the hybrid allocator's
    # 1536-token pages do; columns past the allocation stay 0 (the null block).
    block_table = torch.zeros(b, bt_cols, dtype=torch.int32)
    for row, base in enumerate(base_ids):
        span = min(bt_cols, 12)
        block_table[row, :span] = torch.arange(base, base + span, dtype=torch.int32)
    bt_npu = block_table.npu()

    k_fp8 = torch.zeros(total_blocks, bs, nkv, d, dtype=torch.uint8).npu()
    k_scale = torch.zeros(total_blocks, bs, nkv, d // 64, 2, dtype=torch.uint8).npu()
    v_fp8 = torch.zeros(total_blocks, bs, nkv, d, dtype=torch.uint8).npu()
    v_scale = torch.zeros(total_blocks, bs // 64, nkv, d, 2, dtype=torch.uint8).npu()
    staging = torch.zeros(b + 1, 128, nkv, d, dtype=torch.bfloat16).npu()

    horizon = max(prompt_lens) + steps + 4
    keys = [torch.randn(horizon, nkv, d, dtype=torch.bfloat16).npu() for _ in range(b)]
    values = [torch.randn(horizon, nkv, d, dtype=torch.bfloat16).npu() for _ in range(b)]
    cur = [0] * b

    def slots_for(req, lo, hi):
        pos = torch.arange(lo, hi, device="npu")
        return bt_npu[req, torch.div(pos, bs, rounding_mode="floor")].long() * bs + pos % bs

    def run_step(active, q_lens, bulk):
        """One engine step over `active` requests taking `q_lens` tokens each."""
        key_cat = torch.cat([keys[r][cur[r] : cur[r] + q] for r, q in zip(active, q_lens)])
        val_cat = torch.cat([values[r][cur[r] : cur[r] + q] for r, q in zip(active, q_lens)])
        all_slots = torch.cat([slots_for(r, cur[r], cur[r] + q) for r, q in zip(active, q_lens)])
        kf, ks = _impl_quant_key(key_cat)
        k_fp8.view(-1, nkv, d).index_put_((all_slots,), kf)
        k_scale.view(-1, nkv, d // 64, 2).index_put_((all_slots,), ks)

        new_lens = [cur[r] + q for r, q in zip(active, q_lens)]
        sub_bt = bt_npu[torch.tensor(active).npu()]
        if bulk:
            cu = torch.tensor(q_lens).cumsum(0).tolist()
            _impl_write_value_bulk(staging, v_fp8, v_scale, val_cat, cu, new_lens, sub_bt, len(active))
        else:
            ctx = torch.tensor([cur[r] for r in active], dtype=torch.int32).npu()
            lens = torch.tensor(new_lens, dtype=torch.int32).npu()
            req_ids = torch.arange(len(active)).npu()
            _impl_write_value_decode(
                staging, v_fp8, v_scale, val_cat, req_ids, ctx, ctx, lens, sub_bt, len(active)
            )
        for r, q in zip(active, q_lens):
            cur[r] += q
        torch.npu.synchronize()

    def check(active, tag):
        """Read every active request back with a single decode token."""
        q_all, qs_all, goldens = [], [], []
        for r in active:
            s = cur[r]
            q = torch.randn(1, nq, d, dtype=torch.bfloat16)
            qsc = qk_group_scale(q)
            q_fp8 = quantize_with_scale(q, qsc.repeat_interleave(GROUP, dim=-1)[..., :d])
            q_all.append(q_fp8)
            qs_all.append(pack_last_pairs(fp32_to_e8m0_bytes(qsc, "q")))
            k_cpu, v_cpu = keys[r][:s].cpu(), values[r][:s].cpu()
            ref_ksc = qk_group_scale(k_cpu)
            ref_k = quantize_with_scale(k_cpu, ref_ksc.repeat_interleave(GROUP, dim=-1)[..., :d])
            s_pad = (s + 63) // 64 * 64
            v_pad = torch.nn.functional.pad(v_cpu.float(), (0, 0, 0, 0, 0, s_pad - s))
            ref_vsc = v_group_scale(v_pad)
            ref_v = quantize_with_scale(v_pad, ref_vsc.repeat_interleave(GROUP, dim=0)[:s_pad])[:s]
            goldens.append(
                cpu_golden_one_seq(
                    q_fp8, ref_k, ref_v, qsc, ref_ksc,
                    ref_vsc[: (s + GROUP - 1) // GROUP], softmax_scale, True,
                )
            )
        n_act = len(active)
        qs = torch.cat(qs_all).reshape(n_act, nkv, g, d // 64, 2).permute(1, 0, 2, 3, 4).contiguous()
        npu_kwargs = {
            "q": torch.cat(q_all).npu(),
            "k": k_fp8.view(torch.float8_e4m3fn),
            "v": v_fp8.view(torch.float8_e4m3fn),
            "q_descale": e8m0_npu(qs),
            "k_descale": k_scale.view(torch.float8_e8m0fnu),
            "v_descale": v_scale.view(torch.float8_e8m0fnu),
            "block_table": bt_npu[torch.tensor(active).npu()],
            "cu_seqlens_q": torch.arange(n_act + 1, dtype=torch.int32).npu(),
            "seqused_kv": torch.tensor([cur[r] for r in active], dtype=torch.int32).npu(),
            "softmax_scale": softmax_scale,
            "mask_mode": 3,
            "max_seqlen_q": 1,
            "max_seqlen_kv": max(cur[r] for r in active),
            "layout_q": "TND",
            "layout_q_descale": "N2TGD",
            "layout_kv": "PA_BBND",
            "layout_out": "TND",
        }
        out = call_qfa(npu_kwargs, nq, nkv, d, causal_mask_npu()).float().cpu()
        bad = []
        for i, r in enumerate(active):
            cos = torch.nn.functional.cosine_similarity(
                out[i : i + 1].reshape(-1), goldens[i].float().reshape(-1), dim=0
            ).item()
            if cos < 0.999:
                bad.append((f"req{r}", cur[r], round(cos, 5)))
        lens = [cur[r] for r in active]
        print(f"  [8] {tag} lens={lens}: {'ok' if not bad else f'DIVERGED {bad}'}")
        return not bad

    ok = True
    run_step([0], [prompt_lens[0]], bulk=True)  # step 1: PrefillNoCache, req 0 alone
    ok &= check([0], "after solo prefill")
    # step 2: ChunkedPrefill - req 0 decodes while reqs 1 and 2 prefill
    run_step([0, 1, 2], [1, prompt_lens[1], prompt_lens[2]], bulk=True)
    ok &= check([0, 1, 2], "after mixed batch")
    for step in range(1, steps + 1):  # steps 3+: pure decode
        run_step([0, 1, 2], [1, 1, 1], bulk=False)
        ok &= check([0, 1, 2], f"after decode {step}")
        if not ok:
            break
    return ok


def case_real_schedule() -> bool:
    """The original hand-built geometry: narrow table, block ids from zero."""
    print("== case 8: real engine schedule (solo prefill -> mixed -> decode) ==")
    ok = _run_schedule(6, 2, [0, 2, 4], "8: narrow table, ids from 0")
    print(f"  [SCHEDULE] {'GREEN' if ok else 'RED'}")
    return ok


def case_real_shapes() -> bool:
    """Walk the block-table geometry from the test's toward the engine's.

    Everything about the write side is verified byte-correct on hardware and
    the first batch row reads back fine, so what is left is a per-row indexing
    difference the earlier cases never exercised: the live table is 36 columns
    wide, its ids start well above zero, and it indexes a cache of thousands
    of blocks. Each variant changes exactly one of those.
    """
    print("== case 9: engine block-table geometry ==")
    variants = (
        (6, 2, [0, 2, 4], "9a control (= case 8)"),
        (128, 2, [12, 60, 108], "9b real ids, narrow table"),
        (128, 36, [12, 60, 108], "9c real ids, 36-column table"),
        (2293, 36, [12, 60, 108], "9d real ids, wide table, full-size cache"),
    )
    results = []
    for total_blocks, bt_cols, base_ids, label in variants:
        ok = _run_schedule(total_blocks, bt_cols, base_ids, label)
        results.append((label, ok))
    for label, ok in results:
        print(f"  [9] {label}: {'ok' if ok else 'DIVERGED'}")
    ok = all(o for _, o in results)
    print(f"  [SHAPES] {'GREEN' if ok else 'RED'}")
    return ok


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
        ("C-SHAPES", case_c_decode_shapes),
        ("COMPOSE", case_write_read_composition),
        ("BATCH-STEPS", case_batched_decode_steps),
        ("SCHEDULE", case_real_schedule),
        ("SHAPES", case_real_shapes),
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
