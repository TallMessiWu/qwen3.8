#!/usr/bin/env python3
"""Which V-scale policy costs least when a verify step writes unconfirmed tokens?

The MXFP8 V cache shares one E8M0 scale across 32 tokens, and a verify step has
to write all 1 + num_spec candidates before the attention read that judges them.
Tokens that may yet be rejected therefore raise the scale and re-round every
token already committed to the window. This picks the policy that answers it.

Pure arithmetic about E4M3 and E8M0 - no NPU, no vLLM. It streams a decode
through 64-token windows keeping a cache the way the real write path does
(history is read back out of the cache, never from the originals), so rounding
error compounds exactly as it does in service.

Policies:
  today      scale from the whole window, drafts included. What the branch does.
  confirmed  scale from the confirmed tokens only; drafts clamp into it.
  cap+h      let the drafts raise the scale, but by at most h E8M0 steps.
  restore    `confirmed` for the attention read, then rewrite the window at the
             full scale with the unclamped values once the read is done.
  defer      `confirmed`, plus a one-step raw-V buffer that puts newly confirmed
             tokens back at their true value before the next re-quantize.

Two numbers matter and they are not the same:
  vs-nonspec  how far the KV the first query reads has drifted from what the
              same model without speculation would have held. This is what
              makes a greedy chain diverge, and what SELF_SPEC in the smoke
              test measures.
  vs-truth    how far it has drifted from the original BF16 values. This is
              what the BF16 comparison measures. `today` already matches the
              non-speculative run here - speculation does not make the cache
              less accurate, it rounds it differently by the same magnitude.

Run (seconds, any machine with torch):
    python scripts/debug/sim_qfa_spec_scale_policies.py
"""

import os
import sys

import torch

PACK = 32  # tokens sharing one E8M0 scale
GROUP = 64  # tokens per V window (two packed scale rows)
E4M3_MAX = 448.0
NUM_SPEC = int(os.environ.get("NUM_SPEC", "3"))
T = int(os.environ.get("TOKENS", "512"))
C = int(os.environ.get("CHANNELS", "1024"))  # num_kv_heads * head_size
DEV = "cuda" if torch.cuda.is_available() else "cpu"
POLICIES = ["today", "confirmed", "cap+1", "restore", "defer"]


def e8m0_byte(amax: torch.Tensor) -> torch.Tensor:
    """scale_alg=0: shared exponent taken straight off the block maximum.

    Confirmed against npu_dynamic_mx_quant by probe_qfa_mx_scale_npu.py; the
    conclusions here are about which policy wins, and that ordering does not
    depend on the exact rule.
    """
    byte = torch.where(
        amax > 0,
        torch.floor(torch.log2(amax.clamp(min=1e-45))) - 8 + 127,
        torch.zeros_like(amax),
    )
    return byte.clamp(0, 255)


def quant_pack(vals: torch.Tensor, byte: torch.Tensor) -> torch.Tensor:
    return (vals / torch.exp2(byte - 127)).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)


def dequant_pack(q: torch.Tensor, byte: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * torch.exp2(byte - 127)


def pack_amax(rows: torch.Tensor) -> torch.Tensor:
    """(64, C) -> (2, C), one absolute maximum per packed scale row."""
    return rows.abs().reshape(2, PACK, C).amax(dim=1)


class Cache:
    """The V planes and their packed E8M0 rows, one entry per 64-token window."""

    def __init__(self, n_windows: int):
        self.q = torch.zeros(n_windows, GROUP, C, dtype=torch.float8_e4m3fn, device=DEV)
        self.byte = torch.zeros(n_windows, 2, C, device=DEV)

    def read(self, w: int) -> torch.Tensor:
        out = torch.empty(GROUP, C, device=DEV)
        for p in range(2):
            sl = slice(p * PACK, (p + 1) * PACK)
            out[sl] = dequant_pack(self.q[w, sl], self.byte[w, p])
        return out

    def write(self, w: int, rows: torch.Tensor, byte: torch.Tensor) -> None:
        for p in range(2):
            sl = slice(p * PACK, (p + 1) * PACK)
            self.q[w, sl] = quant_pack(rows[sl], byte[p])
            self.byte[w, p] = byte[p]


def choose_byte(rows: torch.Tensor, confirmed: int, policy: str) -> torch.Tensor:
    """The E8M0 byte each of the window's two packed rows is written at."""
    full = e8m0_byte(pack_amax(rows))
    if policy == "today":
        return full
    conf_rows = rows.clone()
    conf_rows[confirmed:] = 0
    conf = e8m0_byte(pack_amax(conf_rows))
    # A packed row holding no confirmed token has nothing to scale from.
    has_conf = conf > 0
    if policy.startswith("cap+"):
        h = int(policy.split("+")[1])
        return torch.where(has_conf, torch.minimum(full, conf + h), full)
    return torch.where(has_conf, conf, full)


def run(policy: str, vals: torch.Tensor, accepts: list[int]) -> dict:
    n_windows = (T + GROUP - 1) // GROUP
    cache = Cache(n_windows)
    ref = Cache(n_windows)  # the same model without speculation

    ctx, step = 0, 0
    just_written: list[int] = []
    drift, truth_err, ref_err = [], [], []
    while ctx < T:
        k = min(NUM_SPEC, T - ctx - 1)
        end = ctx + 1 + k
        for p in range(ctx, end):  # advance the non-speculative reference
            w, off = p // GROUP, p % GROUP
            rows = ref.read(w)
            rows[off + 1 :] = 0
            rows[off] = vals[p]
            ref.write(w, rows, e8m0_byte(pack_amax(rows)))

        unclamped = {}
        for w in range(ctx // GROUP, (end - 1) // GROUP + 1):
            base, top = w * GROUP, (w + 1) * GROUP
            rows = cache.read(w)
            rows[min(max(end - base, 0), GROUP) :] = 0  # drop rejected drafts
            if policy == "defer":
                back = torch.tensor(
                    [base + i in just_written and base + i < ctx for i in range(GROUP)], device=DEV
                )
                rows = torch.where(back.unsqueeze(1), vals[base:top], rows)
            lo, hi = max(ctx, base), min(end, top)
            rows[lo - base : hi - base] = vals[lo:hi]
            unclamped[w] = rows.clone()
            sub = "confirmed" if policy in ("restore", "defer") else policy
            cache.write(w, rows, choose_byte(rows, min(max(ctx + 1 - base, 0), GROUP), sub))

        # What the first query of this step reads: KV 0..ctx, all confirmed.
        w0, n = ctx // GROUP, ctx % GROUP + 1
        got, want = cache.read(w0)[:n], ref.read(w0)[:n]
        true = vals[w0 * GROUP : w0 * GROUP + n]
        span = want.abs().amax().clamp(min=1e-9)
        # Attention concentrates on recent tokens; weighting separates "wrong
        # in a value nobody reads" from "wrong in the sum the logits come from".
        weight = torch.softmax(torch.linspace(-3, 0, n, device=DEV), dim=0).unsqueeze(1)
        drift.append(float(((got - want) * weight).sum(0).abs().max() / span))
        truth_err.append(float(((got - true) * weight).sum(0).abs().max() / span))
        ref_err.append(float(((want - true) * weight).sum(0).abs().max() / span))

        if policy == "restore":
            # The read is done: put the window back at the scale the unclamped
            # values need, so a crushed byte never becomes history.
            for w, rows in unclamped.items():
                cache.write(w, rows, e8m0_byte(pack_amax(rows)))

        just_written = list(range(ctx, end))
        ctx += min(accepts[step], k) + 1
        step += 1

    committed = torch.stack([cache.read(w) for w in range(n_windows)]).reshape(-1, C)[:T]
    return {
        "drift": max(drift),
        "truth": max(truth_err),
        "ref": max(ref_err),
        "cache": float((committed - vals[:T]).abs().max() / vals[:T].abs().amax()),
        "steps": len(drift),
    }


PROFILES = {
    "flat gaussian": lambda g: torch.randn(T, C, device=DEV, generator=g),
    "per-channel spread 64x": lambda g: torch.randn(T, C, device=DEV, generator=g)
    * torch.exp2(torch.rand(1, C, device=DEV, generator=g) * 6),
    "heavy tail": lambda g: torch.randn(T, C, device=DEV, generator=g)
    / torch.rand(T, C, device=DEV, generator=g).clamp(min=0.05).sqrt(),
    "drifting magnitude 32x": lambda g: torch.randn(T, C, device=DEV, generator=g)
    * torch.exp2(torch.linspace(0, 5, T, device=DEV)).unsqueeze(1),
}


def main() -> int:
    print(f"device={DEV} tokens={T} channels={C} num_spec={NUM_SPEC}")
    worst = {p: {"drift": 0.0, "truth": 0.0, "cache": 0.0} for p in POLICIES}
    ref_worst = 0.0
    for accept_p, label in ((0.0, "all rejected"), (0.7, "70% accepted")):
        print(f"\n=== acceptance: {label}")
        g = torch.Generator(device=DEV).manual_seed(20260827)
        accepts = [
            int(torch.randint(0, NUM_SPEC + 1, (1,), device=DEV, generator=g).item())
            if float(torch.rand(1, device=DEV, generator=g)) < accept_p
            else 0
            for _ in range(T)
        ]
        for pname, make in PROFILES.items():
            vals = make(torch.Generator(device=DEV).manual_seed(7)).float()
            print(f"  -- {pname}")
            for policy in POLICIES:
                r = run(policy, vals, accepts)
                print(
                    f"     {policy:<10} vs-nonspec {r['drift']:.5f}  vs-truth {r['truth']:.5f}  "
                    f"cache {r['cache']:.5f}   (nonspec itself {r['ref']:.5f}, {r['steps']} steps)"
                )
                for key in ("drift", "truth", "cache"):
                    worst[policy][key] = max(worst[policy][key], r[key])
                ref_worst = max(ref_worst, r["ref"])

    print("\n=== worst case across every profile and acceptance rate")
    for policy in POLICIES:
        w = worst[policy]
        print(f"  {policy:<10} vs-nonspec {w['drift']:.5f}  vs-truth {w['truth']:.5f}  cache {w['cache']:.5f}")
    print(f"  {'(nonspec)':<10} vs-truth {ref_worst:.5f}  - the floor any policy is measured against")

    # The policy is only worth adopting if it drifts less than today WITHOUT
    # giving up accuracy against the original values, which is what a policy
    # that clamps drafts into the cache for good does.
    ok = worst["restore"]["drift"] < worst["today"]["drift"] and worst["restore"]["truth"] <= worst["today"]["truth"]
    regress = worst["confirmed"]["truth"] > worst["today"]["truth"]
    print()
    print(f"  restore drifts {worst['today']['drift'] / max(worst['restore']['drift'], 1e-12):.1f}x less than today")
    print(f"  confirmed-only alone {'regresses' if regress else 'does not regress'} accuracy against the true values")
    if not ok:
        print("[RED] restore is not the better policy on this data - re-open the design")
        return 1
    print("[GREEN] restore: less drift than today, same accuracy against the true values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
