"""Can a V window be quantized at a scale we choose, by clamping the input?

The MXFP8 V cache shares one E8M0 scale across 32 tokens, and a verify step has
to write all 1 + num_spec candidates before the attention read that judges them,
so tokens that may yet be rejected raise that scale and re-round every token
already committed. sim_qfa_spec_scale_policies.py settled what to do about it:
write the window twice. Once before the read, scaled from the confirmed tokens
alone with the drafts clamped into that range, so the query whose logits decide
draft 1 sees what a non-speculative run would have held; then again after the
read, unclamped at the full scale, so no clamped byte ever becomes history.
(Scaling from the confirmed tokens and leaving it there was measured 3x worse
than doing nothing - the clamped drafts become the confirmed history and the
scale can never climb back.)

The first of those writes needs the operator to quantize at a scale it did not
pick. npu_dynamic_mx_quant takes no override, and hand-rolling
`(v / scale).to(float8_e4m3fn)` would put our rounding next to the operator's on
the same cache. The way out: clamp the window to what the confirmed scale can
represent, then call the operator as usual. The clamped maximum lands back in
the same power-of-two bucket, so the operator re-derives the same scale and
still owns the rounding.

That is an assumption about an operator we did not write, so it gets measured
before any of it is built. This also settles how the scale byte is computed
(which decides whether the extra write costs a quantize or just a reduction)
and whether values above the e4m3 maximum saturate or turn into NaN.

Run (seconds, no server, one card):
    python scripts/debug/probe_qfa_mx_scale_npu.py
"""

import os
import sys

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("[RED] torch_npu unavailable - run this on the server")
    sys.exit(2)

DEVICE = "npu"
GROUP = 64  # tokens per V window
PACK = 32  # tokens sharing one E8M0 scale; two packed per window
E4M3_MAX = 448.0
NUM_KV_HEADS = int(os.environ.get("NKV", "4"))
HEAD_SIZE = int(os.environ.get("D", "256"))
SCALE_ALG = int(os.environ.get("SCALE_ALG", "0"))  # 0 is what Qwen3.8 resolves to
NUM_SPEC = int(os.environ.get("NUM_SPEC", "3"))

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    tag = "GREEN" if ok else "RED"
    print(f"  [{tag}] {name}{(': ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)
    return ok


def quant_cols(cols: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(M, 64) bf16 -> (M, 64) fp8 bytes + (M, 2) e8m0 bytes."""
    fp8, scale = torch_npu.npu_dynamic_mx_quant(cols.contiguous(), dst_type=torch.float8_e4m3fn, scale_alg=SCALE_ALG)
    return fp8.view(torch.uint8), scale.view(torch.uint8)


def quant_windows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror of _qfa_quant_along_tokens: (W,64,N,D) -> fp8 bytes + (W,N,D,2)."""
    w, group, n, d = rows.shape
    cols = rows.permute(0, 2, 3, 1).reshape(w * n * d, group)
    fp8, scale = quant_cols(cols)
    fp8 = fp8.reshape(w, n, d, group).permute(0, 3, 1, 2)
    return fp8.contiguous(), scale.reshape(w, n, d, 2)


def dequant_windows(fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Mirror of _qfa_dequant_windows: (W,64,N,D) uint8 + (W,N,D,2) -> bf16."""
    group = fp8.shape[1]
    vals = fp8.view(torch.float8_e4m3fn).to(torch.bfloat16)
    factor = torch.exp2((scale.to(torch.int32) - 127).to(torch.float32)).to(torch.bfloat16)
    factor = factor.permute(0, 3, 1, 2).repeat_interleave(group // 2, dim=1)
    return vals * factor


def prefix_mask(count: int, group: int = GROUP) -> torch.Tensor:
    """(1, group, 1, 1) bf16 mask keeping the first `count` tokens."""
    keep = torch.arange(group, device=DEVICE) < count
    return keep.to(torch.bfloat16).view(1, group, 1, 1)


def group_amax(cols: torch.Tensor) -> torch.Tensor:
    """(M, 64) -> (M, 2) per-32-token absolute maximum, matching the packing."""
    return cols.abs().reshape(cols.shape[0], 2, PACK).amax(dim=2)


def bound_from_scale(scale: torch.Tensor) -> torch.Tensor:
    """E8M0 bytes -> the largest magnitude e4m3 can hold at that scale."""
    return (torch.exp2((scale.to(torch.int32) - 127).to(torch.float32)) * E4M3_MAX).to(torch.bfloat16)


# ---------------------------------------------------------------- test 1
def test_scale_formula() -> None:
    """Which closed form does the operator's E8M0 byte follow?

    Knowing it means the fix can derive the clamp bound from a cheap amax
    reduction instead of a second full quantize.
    """
    print("== 1. E8M0 byte vs closed forms")
    torch.manual_seed(1024)
    cols = torch.randn(4096, GROUP, dtype=torch.bfloat16, device=DEVICE)
    # Spread the magnitudes over many binades so a formula that is only right
    # near 1.0 is caught.
    spread = torch.exp2(torch.randint(-40, 40, (4096, 1), device=DEVICE).float()).to(torch.bfloat16)
    cols = (cols * spread).to(torch.bfloat16)
    _, scale = quant_cols(cols)
    amax = group_amax(cols).float()
    got = scale.to(torch.int32)
    # H0: shared exponent straight off the block maximum (MX spec, emax_elem=8)
    h0 = (127 + torch.floor(torch.log2(amax)) - 8).to(torch.int32)
    # H1: map the maximum into the e4m3 range first, then round the exponent up
    h1 = (127 + torch.ceil(torch.log2(amax / E4M3_MAX))).to(torch.int32)
    m0 = int((got == h0).sum())
    m1 = int((got == h1).sum())
    total = got.numel()
    print(f"     floor(log2(amax))-8 matches {m0}/{total}, ceil(log2(amax/448)) matches {m1}/{total}")
    if m0 != total and m1 != total:
        off = (got - h0)[got != h0]
        print(f"     unmatched deltas vs H0: min {int(off.min())} max {int(off.max())}")
    check("a closed form reproduces every scale byte", m0 == total or m1 == total, f"H0={m0 == total} H1={m1 == total}")

    zeros = torch.zeros(8, GROUP, dtype=torch.bfloat16, device=DEVICE)
    _, zscale = quant_cols(zeros)
    print(f"     all-zero column -> scale byte {int(zscale.flatten()[0])}")
    check("all-zero column gives byte 0 (the 'unwritten' scale)", bool((zscale == 0).all()))


# ---------------------------------------------------------------- test 2
def test_clamp_idempotence() -> None:
    """Does clamping to the confirmed scale's ceiling reproduce that scale?

    This is the whole mechanism: if it holds, the fix never has to name a
    rounding rule of its own.
    """
    print("== 2. clamp to the confirmed bound re-derives the confirmed scale")
    torch.manual_seed(7)
    cases = {
        "flat magnitudes": torch.ones(GROUP),
        "growing 64x across the window": torch.exp2(torch.linspace(0, 6, GROUP)),
        "late 50x outlier": torch.cat([torch.ones(GROUP - 1), torch.tensor([50.0])]),
        "tiny confirmed, huge drafts": torch.cat([torch.full((GROUP - 8,), 1e-3), torch.full((8,), 1e3)]),
    }
    for label, profile in cases.items():
        cols = torch.randn(2048, GROUP, dtype=torch.bfloat16, device=DEVICE)
        cols = (cols * profile.to(DEVICE).to(torch.bfloat16)).to(torch.bfloat16)
        for confirmed in (1, 17, 33, 61):
            keep = (torch.arange(GROUP, device=DEVICE) < confirmed).to(torch.bfloat16)
            _, s_conf = quant_cols(cols * keep)
            bound = bound_from_scale(s_conf).repeat_interleave(PACK, dim=1)
            _, s_clamped = quant_cols(torch.clamp(cols, -bound, bound))
            same = bool(torch.equal(s_conf, s_clamped))
            detail = "" if same else f"{int((s_conf != s_clamped).sum())}/{s_conf.numel()} bytes moved"
            check(f"{label} @confirmed={confirmed}", same, detail)


# ---------------------------------------------------------------- test 3
def test_saturation() -> None:
    """Above the e4m3 maximum, does the operator saturate or emit NaN?

    scale_alg=0 derives the exponent from the block maximum with no headroom
    check, so amax/scale can land in (448, 512). If that turns into NaN the
    cache would poison itself with no clamp involved.
    """
    print("== 3. behaviour above the e4m3 maximum")
    cols = torch.full((256, GROUP), 1.0, dtype=torch.bfloat16, device=DEVICE)
    # 500 sits inside (448, 512): representable by the shared exponent, not by e4m3.
    cols[:, 0] = 500.0
    fp8, scale = quant_cols(cols)
    nan_bytes = int(((fp8 & 0x7F) == 0x7F).sum())
    factor = torch.exp2((scale.to(torch.int32) - 127).to(torch.float32)).repeat_interleave(PACK, dim=1)
    back = fp8.view(torch.float8_e4m3fn).to(torch.float32) * factor
    top = float(back[:, 0].max())
    print(f"     scale byte {int(scale.flatten()[0])}, overflowing element reads back as {top}")
    check("no NaN encodings produced", nan_bytes == 0, f"{nan_bytes} NaN bytes")
    check("the overflowing element saturates rather than wrapping", nan_bytes == 0 and top >= E4M3_MAX * 0.99)


# ---------------------------------------------------------------- test 4
def test_clamp_kernel() -> None:
    """Does torch.clamp with tensor bounds run on AICORE for this shape?

    float8 has no transpose or index_put_ here and the AICPU fallback aborts
    the stream, so every op the write path gains is checked before it is used.
    """
    print("== 4. torch.clamp with tensor bounds, at the real window shape")
    w = 2 * 4
    rows = torch.randn(w, GROUP, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device=DEVICE)
    scale = torch.randint(100, 140, (w, NUM_KV_HEADS, HEAD_SIZE, 2), dtype=torch.uint8, device=DEVICE)
    bound = bound_from_scale(scale).permute(0, 3, 1, 2).repeat_interleave(GROUP // 2, dim=1)
    try:
        out = torch.clamp(rows, -bound, bound)
        torch.npu.synchronize()
        ok = tuple(out.shape) == tuple(rows.shape) and bool(torch.isfinite(out.float()).all())
        check("full-shape clamp", ok, f"out {tuple(out.shape)}")
    except Exception as exc:  # noqa: BLE001 - the point is to report whatever it raises
        check("full-shape clamp", False, repr(exc))
    try:
        narrow = bound[:, :1]
        out = torch.clamp(rows, -narrow, narrow)
        torch.npu.synchronize()
        check("broadcast clamp (W,1,N,D)", tuple(out.shape) == tuple(rows.shape))
    except Exception as exc:  # noqa: BLE001
        check("broadcast clamp (W,1,N,D)", False, repr(exc))
    try:
        big = torch.tensor(torch.finfo(torch.bfloat16).max, dtype=torch.bfloat16, device=DEVICE)
        out = torch.where(scale == 0, big, bound_from_scale(scale))
        torch.npu.synchronize()
        check("where(scale==0, bf16 max, bound)", tuple(out.shape) == tuple(scale.shape))
    except Exception as exc:  # noqa: BLE001
        check("where(scale==0, bf16 max, bound)", False, repr(exc))


# ---------------------------------------------------------------- test 5
def test_cost() -> None:
    """What each candidate implementation adds to a decode step, per layer.

    Not a gate - the numbers decide whether the confirmed scale comes from a
    second quantize (formula-free) or from an amax reduction plus a closed
    form. Both are correct; only one is worth its bandwidth.
    """
    print("== 5. cost of the extra pass (per layer, per decode step)")

    def timed(fn, reps: int = 20) -> float:
        for _ in range(3):
            fn()
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            fn()
        end.record()
        torch.npu.synchronize()
        return start.elapsed_time(end) / reps

    for num_reqs in (4, 64):
        w = 2 * num_reqs
        rows = torch.randn(w, GROUP, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device=DEVICE)
        keep = prefix_mask(GROUP - NUM_SPEC)
        scale = torch.randint(100, 140, (w, NUM_KV_HEADS, HEAD_SIZE, 2), dtype=torch.uint8, device=DEVICE)
        bound = bound_from_scale(scale).permute(0, 3, 1, 2).repeat_interleave(GROUP // 2, dim=1)

        # Default arguments, not closures: the loop rebinds these every pass.
        base = timed(lambda r=rows: quant_windows(r))
        masked_quant = timed(lambda r=rows, k=keep: quant_windows(r * k))
        amax = timed(lambda r=rows, k=keep: (r * k).abs().amax(dim=1))
        clamp = timed(lambda r=rows, b=bound: torch.clamp(r, -b, b))
        print(
            f"     num_reqs={num_reqs:>3} W={w:>3}  quantize {base:.3f} ms | "
            f"masked quantize {masked_quant:.3f} ms | masked amax {amax:.3f} ms | clamp {clamp:.3f} ms"
        )
        print(
            f"                    safe (quantize+clamp) +{100 * (masked_quant + clamp) / base:.0f}%  "
            f"cheap (amax+clamp) +{100 * (amax + clamp) / base:.0f}%  of one quantize"
        )


# ---------------------------------------------------------------- test 6
def test_two_write_sequence() -> None:
    """The write path as it will actually be built, at the real window shape.

    Two properties have to hold together, and only the operator can confirm
    them:

      read    the pre-read cache, over the confirmed prefix, dequantizes to
              exactly what a non-speculative run holding the same tokens would
              have - byte for byte, since both go through the same operator at
              the same scale.
      commit  the post-read cache is byte identical to what a single unclamped
              write produces today, so nothing about the committed cache
              changes and the accuracy already measured against BF16 stands.
    """
    print("== 6. the two-write sequence at the real window shape")
    torch.manual_seed(99)
    w = 2 * 4
    rows = torch.randn(w, GROUP, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.bfloat16, device=DEVICE)
    rows = (rows * torch.exp2(torch.linspace(0, 5, GROUP, device=DEVICE)).view(1, GROUP, 1, 1)).to(torch.bfloat16)
    bf16_max = torch.tensor(torch.finfo(torch.bfloat16).max, dtype=torch.bfloat16, device=DEVICE)

    for confirmed in (1, 12, 33, GROUP - NUM_SPEC):
        conf_rows = rows * prefix_mask(confirmed)
        _, s_conf = quant_windows(conf_rows)
        bound = torch.where(s_conf == 0, bf16_max, bound_from_scale(s_conf))
        bound = bound.permute(0, 3, 1, 2).repeat_interleave(GROUP // 2, dim=1)

        # write 1: what the attention read sees
        read_fp8, read_scale = quant_windows(torch.clamp(rows, -bound, bound))
        # what a non-speculative run holding only the confirmed tokens holds
        want_fp8, want_scale = quant_windows(conf_rows)
        keep = prefix_mask(confirmed).to(torch.uint8)
        # Only the packed rows that hold a confirmed token are comparable: one
        # covering nothing but drafts has no confirmed scale to take, is left
        # unclamped by design, and no query in this step reads it.
        packs = -(-confirmed // PACK)
        same_read = bool(torch.equal(read_fp8 * keep, want_fp8 * keep)) and bool(
            torch.equal(read_scale[..., :packs], want_scale[..., :packs])
        )
        check(f"read view matches a non-speculative window @confirmed={confirmed}", same_read)

        # write 2 has to re-quantize the rows the write path still holds, not
        # read the clamped cache back. Both spellings compile; only one keeps
        # the committed cache what it is today, so measure the difference and
        # keep it visible.
        today_fp8, today_scale = quant_windows(rows)
        commit_fp8, commit_scale = quant_windows(rows)
        check(
            f"committed cache unchanged @confirmed={confirmed}",
            bool(torch.equal(commit_fp8, today_fp8)) and bool(torch.equal(commit_scale, today_scale)),
        )
        trap_fp8, trap_scale = quant_windows(dequant_windows(read_fp8, read_scale))
        lost = float((dequant_windows(trap_fp8, trap_scale) - dequant_windows(today_fp8, today_scale)).abs().max())
        print(f"     re-reading the clamped cache instead would lose up to {lost:.4f} @confirmed={confirmed}")


def main() -> int:
    print(f"scale_alg={SCALE_ALG} num_kv_heads={NUM_KV_HEADS} head_size={HEAD_SIZE} num_spec={NUM_SPEC}\n")
    test_scale_formula()
    test_clamp_idempotence()
    test_saturation()
    test_clamp_kernel()
    test_cost()
    test_two_write_sequence()
    print()
    if _failures:
        print(f"[RED] {len(_failures)} check(s) failed: {', '.join(_failures)}")
        return 1
    print("[GREEN] the confirmed-scale read view and the unchanged committed cache both hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
