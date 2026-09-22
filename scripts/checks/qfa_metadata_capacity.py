#!/usr/bin/env python3
"""Does quant_flash_attn_metadata allocate enough for the plan it then writes?

Minimal reproducer for the AICPU abort that kills a PD-disaggregated P node on
the newer ops-transformer delivery. Only the metadata operator is involved --
no q/k/v, no KV cache, no model -- because the operator takes nothing but
lengths and attrs. That makes this small enough to hand to the operator team.

The defect, as read out of the sources:

  Python wrapper (cann_ops_transformer):
      max_schedule_size = _calculate_max_schedule_size(batch_size, num_heads_kv)
      output = torch.empty((2, max_schedule_size), dtype=torch.int32)
    and its own docstring states the assumption -- "dim0 按 sectionNum 最坏值
    (batch*num_heads_kv) 动态计算".

  AICPU kernel (quant_flash_attn_metadata_aicpu.cpp):
      baseInfo.kvHeadNum = isDecode ? numHeadsKv_ : numHeadsQ_;   // isDecode = layout_q_descale == "N2TGD"
    and CalcGridInfoSection's inner loop counts to baseInfo.GetKvHeadNum(), so
    under TND (prefill) sectionNum's worst case is batch * num_heads_Q.

  Under GQA the two disagree by G = num_heads_q / num_heads_kv, the buffer is G
  times too small, and FaMetaData::Clear() runs off the end. The abort takes the
  device with it: every later op on the stream fails with 507018 and the Python
  traceback lands on whatever ran next (GDN, SwiGlu, Cumsum), never on QFA.

Why the older delivery never showed it: section splitting is gated on
``param.l2Byte``, which used to be 0 -- so sectionNum was pinned at 1, the need
was a flat 16 + (aic + aiv) * 16 bytes, and the wrapper's 4096-byte alignment
covered it no matter what the heads were. MXFP8 now turns FlashDecode on with
l2Byte = 96MB, splitting starts once a single head's tokens exceed
l2Byte / aic_num, and the shortfall surfaces.

Three phases:

  BUDGET   Pure arithmetic against the installed wrapper: what it allocates vs
           what the kernel needs, for prefill (TND) and decode (N2TGD). Touches
           no NPU, so it prints a verdict even on a host that cannot run the op.
  SCAN     Calls the operator for real, one sequence length at a time, TND, each
           in its own subprocess because an AICPU abort poisons the device.
           Reports the first length that dies.
  CONTROL  The single-variable check: the exact (batch, seq_len) that died under
           TND, rerun with layout_q_descale="N2TGD" and nothing else changed. If
           TND dies and N2TGD survives, the head-count mismatch is the cause and
           not the sequence length, the batch, or capacity in general.

Exit code is the point: 0 on a delivery that holds up (run it on the OLD
package to get your baseline), non-zero once the shortfall reproduces.

Usage (inside the serving container, one NPU is enough):
    python3 scripts/checks/qfa_metadata_capacity.py
    python3 scripts/checks/qfa_metadata_capacity.py --num-heads-q 4 --num-heads-kv 1
    python3 scripts/checks/qfa_metadata_capacity.py --batch 8 --seq-lens 1024,4096,16384
    python3 scripts/checks/qfa_metadata_capacity.py --phases BUDGET   # no NPU needed

Defaults are Qwen3.5-397B per rank at TP8: num_heads_q=4, num_heads_kv=1 (the
model's 2 KV heads are replicated up to 1 per rank once TP exceeds them),
head_dim=256 -- so G=4.
"""

from __future__ import annotations

import argparse
import inspect
import subprocess
import sys

# Kernel-side layout constants (quant_flash_attn_metadata.h).
METADATA_STRIDE_FALLBACK = 16  # FA_METADATA_SIZE == FD_METADATA_SIZE == 16
AIC_FALLBACK, AIV_FALLBACK = 36, 72  # A5, as the operator's own example hardcodes
ALIGN_BYTES = 4096

QUANT_MODE_MXFP8 = 1
MASK_MODE_CAUSAL = 3
LAYOUT_TND = "TND"
LAYOUT_N2TGD = "N2TGD"
LAYOUT_PA_NZ = "PA_NZ"


def load_wrapper():
    """Return (metadata_op, its module) -- the module carries the private helpers."""
    from cann_ops_transformer.ops import quant_flash_attn_metadata as op

    return op, inspect.getmodule(op) or sys.modules.get(op.__module__)


def wrapper_internals(mod):
    """Read the wrapper's own allocation inputs, falling back to kernel constants.

    A delivery that does not expose these (the older one may not) still gets a
    budget table, just one computed from the constants the kernel hardcodes --
    flagged as such, because then it is a model of the wrapper, not the wrapper.
    """
    stride = getattr(mod, "METADATA_STRIDE", None)
    core_fn = getattr(mod, "_get_core_nums", None)
    alloc_fn = getattr(mod, "_calculate_max_schedule_size", None)
    exact = stride is not None and core_fn is not None and alloc_fn is not None
    if core_fn is not None:
        try:
            aic, aiv = core_fn()
        except Exception as exc:  # noqa: BLE001 -- any failure just means fall back
            print(f"  [warn] _get_core_nums() raised {type(exc).__name__}: {exc}")
            aic, aiv = AIC_FALLBACK, AIV_FALLBACK
    else:
        aic, aiv = AIC_FALLBACK, AIV_FALLBACK
    return {
        "stride": stride if stride is not None else METADATA_STRIDE_FALLBACK,
        "aic": aic,
        "aiv": aiv,
        "alloc_fn": alloc_fn,
        "exact": exact,
    }


def model_alloc(internals, batch: int, heads: int) -> int:
    """int32 elements the wrapper allocates, as its formula computes it."""
    stride, aic, aiv = internals["stride"], internals["aic"], internals["aiv"]
    size = stride + aic * stride * batch * heads + aiv * stride * batch * heads
    return ((size + ALIGN_BYTES - 1) // ALIGN_BYTES) * ALIGN_BYTES


def kernel_need(internals, section_num: int) -> int:
    """int32 elements the AICPU kernel writes: 16 + sectionNum * (aic + aiv) * 16."""
    stride, aic, aiv = internals["stride"], internals["aic"], internals["aiv"]
    return stride + section_num * (aic + aiv) * stride


def split_threshold_tokens(internals, head_dim: int, l2_bytes: int = 96 * 1024 * 1024) -> int:
    """Sequence length past which CalcGridInfoSection starts splitting.

    Its bail-out is ``maxSingleHeadTokenCost <= l2Byte / aicCoreMaxNum`` with
    ``singleHeadCost = S * head_dim * 1 * 2 + S * (head_dim + head_dim) * 1``
    for FP8 (one byte per element), i.e. 4 * head_dim * S.
    """
    per_token = 4 * head_dim
    return (l2_bytes // internals["aic"]) // per_token


def phase_budget(internals, batch: int, nq: int, nkv: int, head_dim: int) -> bool:
    """Allocated vs needed, prefill and decode. True when prefill fits."""
    group = nq // nkv if nkv else 0
    alloc = model_alloc(internals, batch, nkv)
    need_prefill = kernel_need(internals, batch * nq)
    need_decode = kernel_need(internals, batch * nkv)
    source = "the installed wrapper" if internals["exact"] else "kernel constants (wrapper helpers not exposed)"
    print(f"  formula inputs from {source}: stride={internals['stride']} aic={internals['aic']} aiv={internals['aiv']}")
    if internals["alloc_fn"] is not None:
        actual = internals["alloc_fn"](batch, nkv)
        print(f"  _calculate_max_schedule_size({batch}, {nkv}) = {actual}")
        if actual != alloc:
            print(f"  [warn] model says {alloc}; using the wrapper's own value")
            alloc = actual
    print(f"  batch={batch} num_heads_q={nq} num_heads_kv={nkv} G={group}")
    print(f"    allocated (wrapper, uses num_heads_kv) : {alloc}")
    print(f"    needed, decode  (sectionNum<={batch}*{nkv}={batch * nkv}) : {need_decode}"
          f"  {'fits' if need_decode <= alloc else 'SHORT'}")
    print(f"    needed, prefill (sectionNum<={batch}*{nq}={batch * nq}) : {need_prefill}"
          f"  {'fits' if need_prefill <= alloc else 'SHORT'}")
    threshold = split_threshold_tokens(internals, head_dim)
    print("  NOTE: the needs above are WORST CASE -- they assume sectionNum reaches")
    print("    batch * head_count. A delivery with param.l2Byte == 0 pins sectionNum")
    print("    at 1 whatever the heads are, so it never gets there. That is why the")
    print("    older package survives an arithmetic shortfall; only SCAN settles it.")
    print("  sections only split once a single head's tokens exceed l2Byte/aic,")
    print(f"    i.e. seq_len > ~{threshold} at head_dim={head_dim}. Below that sectionNum")
    print(f"    stays 1 (need {kernel_need(internals, 1)}) and nothing overruns.")
    ok = need_prefill <= alloc
    if ok:
        verdict = "GREEN -- prefill fits"
    else:
        verdict = (f"RED -- prefill is short by {need_prefill - alloc} int32 "
                   f"({need_prefill / alloc:.1f}x over)")
    print(f"  [BUDGET] {verdict}")
    return ok


def probe_once(nq, nkv, head_dim, batch, seq_len, layout_q_descale) -> int:
    """Child process: one metadata call. Prints the plan shape, or dies trying."""
    import torch
    import torch_npu  # noqa: F401

    op, _ = load_wrapper()
    cu = torch.tensor([i * seq_len for i in range(batch + 1)], dtype=torch.int32).npu()
    kv = torch.full((batch,), seq_len, dtype=torch.int32).npu()
    kwargs = {
        "cu_seqlens_q": cu,
        "cu_seqlens_kv": None,
        "seqused_q": None,
        "seqused_kv": kv,
        "max_seqlen_q": seq_len,
        "max_seqlen_kv": -1,
        "head_dim_v": head_dim,
        "mask_mode": MASK_MODE_CAUSAL,
        "win_left": -1,
        "win_right": -1,
        "layout_q": LAYOUT_TND,
        "layout_q_descale": layout_q_descale,
        "layout_kv": LAYOUT_PA_NZ,
        "layout_out": LAYOUT_TND,
        "is_grad_enabled": False,
    }
    # An older delivery declares neither of the two newest attrs (and does take a
    # v_descale). Send only what this one declares; the point of the run is the
    # capacity, not the signature -- qfa_op_contract.py covers that.
    declared = set(inspect.signature(op).parameters)
    kwargs = {k: v for k, v in kwargs.items() if k in declared}
    out = op(nq, nkv, head_dim, QUANT_MODE_MXFP8, **kwargs)
    torch.npu.synchronize()
    print(f"OK shape={tuple(out.shape)} numel={out.numel()} dtype={out.dtype}")
    return 0


def run_probe(args, seq_len: int, layout: str, timeout: int) -> tuple[bool, str]:
    """Parent side: run one probe in a fresh process. Returns (survived, detail)."""
    cmd = [
        sys.executable, __file__, "--probe",
        "--num-heads-q", str(args.num_heads_q), "--num-heads-kv", str(args.num_heads_kv),
        "--head-dim", str(args.head_dim), "--batch", str(args.batch),
        "--probe-seq-len", str(seq_len), "--probe-layout", layout,
    ]
    try:
        # check=False on purpose: a non-zero exit IS the signal we are after.
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    if r.returncode == 0:
        line = next((ln for ln in r.stdout.splitlines() if ln.startswith("OK ")), "")
        return True, line or "survived"
    tail = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()]
    hint = next((ln.strip() for ln in tail if "QuantFlashAttnMetadata" in ln or "aicpu" in ln.lower()), "")
    return False, f"exit={r.returncode} {hint or (tail[-1].strip() if tail else '')}"[:220]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-heads-q", type=int, default=4, help="per rank (default 397B@TP8: 4)")
    parser.add_argument("--num-heads-kv", type=int, default=1, help="per rank (default 397B@TP8: 1)")
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-lens", default="512,2048,4096,8192,16384,32768",
                        help="comma-separated lengths to scan, ascending")
    parser.add_argument("--phases", default="BUDGET,SCAN,CONTROL")
    parser.add_argument("--timeout", type=int, default=180, help="seconds per probe")
    # internal: one probe in a child process
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe-seq-len", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--probe-layout", default=LAYOUT_TND, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.probe:
        return probe_once(args.num_heads_q, args.num_heads_kv, args.head_dim,
                          args.batch, args.probe_seq_len, args.probe_layout)

    phases = [p.strip().upper() for p in args.phases.split(",") if p.strip()]
    seq_lens = [int(s) for s in args.seq_lens.split(",") if s.strip()]
    print("=== QFA metadata capacity: does the wrapper allocate what the kernel writes? ===")
    print(f"num_heads_q={args.num_heads_q} num_heads_kv={args.num_heads_kv} "
          f"head_dim={args.head_dim} batch={args.batch}")

    try:
        _op, mod = load_wrapper()
    except ImportError as exc:
        print(f"[RED] cannot import cann_ops_transformer: {exc}")
        return 1
    print(f"wrapper: {getattr(mod, '__name__', '?')}")
    internals = wrapper_internals(mod)

    budget_ok = None
    if "BUDGET" in phases:
        print("\n== BUDGET ==")
        budget_ok = phase_budget(internals, args.batch, args.num_heads_q,
                                 args.num_heads_kv, args.head_dim)

    first_bad = None
    if "SCAN" in phases:
        print("\n== SCAN (layout_q_descale=TND, one subprocess per length) ==")
        for seq_len in seq_lens:
            survived, detail = run_probe(args, seq_len, LAYOUT_TND, args.timeout)
            print(f"  seq_len={seq_len:<7} {'ok  ' if survived else 'DIED'} {detail}")
            if not survived and first_bad is None:
                first_bad = seq_len
                break
        if first_bad is None:
            print("  [SCAN] every length survived on this delivery")
        else:
            threshold = split_threshold_tokens(internals, args.head_dim)
            print(f"  [SCAN] first failure at seq_len={first_bad}; sections are predicted to")
            print(f"         start splitting past ~{threshold}, so this is "
                  f"{'consistent' if first_bad > threshold else 'EARLIER THAN PREDICTED'}")

    control_ok = None
    if "CONTROL" in phases and first_bad is not None:
        print(f"\n== CONTROL (same batch={args.batch} seq_len={first_bad}, only layout_q_descale differs) ==")
        survived, detail = run_probe(args, first_bad, LAYOUT_N2TGD, args.timeout)
        control_ok = survived
        print(f"  N2TGD (decode template) {'ok  ' if survived else 'DIED'} {detail}")
        if survived:
            print("  [CONTROL] TND dies where N2TGD lives on identical lengths -- the")
            print("            shortfall follows the head count the kernel picks by layout")
            print("            (numHeadsQ under TND), not the sequence length or the batch.")
        else:
            print("  [CONTROL] N2TGD dies too, so this is NOT the TND/decode head-count")
            print("            mismatch alone -- capacity is short for decode as well.")
    elif "CONTROL" in phases:
        print("\n== CONTROL ==\n  skipped: nothing failed under TND")

    print("\n=== verdict ===")
    # Measurement outranks arithmetic. BUDGET is a worst case and the older
    # delivery never reaches it (sectionNum pinned at 1), so a BUDGET RED with a
    # clean SCAN is that package being fine in practice -- report it as such, and
    # say the risk is latent. Only when SCAN did not run does BUDGET decide.
    scan_ran = "SCAN" in phases
    if scan_ran and first_bad is None:
        print("[GREEN] every probed length survived: this delivery writes within what it allocates.")
        if budget_ok is False:
            print("        (BUDGET's worst case does not fit, but this package never reaches it --")
            print("         sectionNum stays at 1. The shortfall is latent here, not active.)")
        print("        Keep this output as the baseline and rerun after the package swap.")
        return 0
    if not scan_ran:
        if budget_ok is not False:
            print("[GREEN] budget fits (arithmetic only -- rerun with SCAN to confirm on device).")
            return 0
        print("[RED] budget shortfall (arithmetic only -- rerun with SCAN to confirm on device).")
        alloc = model_alloc(internals, args.batch, args.num_heads_kv)
        need = kernel_need(internals, args.batch * args.num_heads_q)
        print(f"      allocates {alloc} int32 for a worst-case plan of {need}")
        return 1
    print("[RED] metadata capacity shortfall reproduced.")
    if budget_ok is False:
        alloc = model_alloc(internals, args.batch, args.num_heads_kv)
        need = kernel_need(internals, args.batch * args.num_heads_q)
        print(f"      budget: allocates {alloc} int32 for a plan that needs {need}")
    if first_bad is not None:
        print(f"      first death: seq_len={first_bad} under layout_q_descale=TND")
    if control_ok:
        print("      same length under N2TGD survives -> the layout-dependent head count is the cause")
    print("      fix belongs in the wrapper: _calculate_max_schedule_size should size")
    print("      with num_heads_q when layout_q_descale != \"N2TGD\", matching")
    print("      baseInfo.kvHeadNum = isDecode ? numHeadsKv : numHeadsQ in the AICPU kernel.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
