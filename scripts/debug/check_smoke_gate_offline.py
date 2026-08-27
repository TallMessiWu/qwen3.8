#!/usr/bin/env python3
"""Replay measured 27B numbers through the smoke test's gate, offline.

No NPU and no model: it feeds smoke_qfa_decode_compare's own report_numerics
the logprobs recorded on hardware, so the gate can be changed without burning
a five-minute run to find out it now passes everything - or nothing.

The three prompts are the 2026-08-27 NUM_SPEC=0 run: prompt 0 diverged at step
47 where the QFA top1-top2 gap was exactly 0.0000, prompt 1 never diverged in
34 steps despite having the largest delta of the three, prompt 2 diverged at
step 2 on a symmetric +-0.2500 flip. All three are quantization noise and must
read GREEN. The sanity cases then pin down what each gate does and does not
catch: a flat prompt where overlap alone would false-RED, the measured MTP
step that has to stay RED, a prediction of what that step should look like
once the verify write is fixed, and an outright broken cache.

Run: python scripts/debug/check_smoke_gate_offline.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoke_qfa_decode_compare as s


def topk(pick_tid, pick_lp, rival_tid, rival_lp, filler_base, filler_shift=0):
    """One step's top-10: the two contenders plus eight also-rans.

    filler_shift renames some of the also-rans, which is how a step with
    churning top-k membership but a stable winner is expressed.
    """
    step = {str(pick_tid): [pick_lp, f"tok{pick_tid}"], str(rival_tid): [rival_lp, f"tok{rival_tid}"]}
    for j in range(8):
        step[str(900 + j + filler_shift)] = [filler_base - j, f"f{j}"]
    return step


def run(
    name,
    chosen,
    deltas,
    div_base,
    div_qfa,
    div_pick_b,
    div_pick_q,
    gate=None,
    rival_lp=-2.0,
    q_shift=0,
):
    """chosen = tokens both runs picked before divergence; deltas = their shifts.

    rival_lp sets how far ahead the winner is - raise it towards the winner to
    make every step a near-tie. q_shift churns the QFA side's also-rans.
    """
    if gate is not None:
        s.DELTA_GATE = gate
    b_steps, q_steps, b_ids, q_ids = [], [], [], []
    for i, (tid, d) in enumerate(zip(chosen, deltas)):
        b_steps.append(topk(tid, -0.10 - i * 0.01, 800 + i, rival_lp, -3.0))
        q_steps.append(topk(tid, -0.10 - i * 0.01 - d, 800 + i, rival_lp, -3.0, q_shift))
        b_ids.append(tid)
        q_ids.append(tid)
    if div_base is not None:
        b_steps.append(topk(div_pick_b, div_base[0], div_pick_q, div_base[1], -3.0))
        q_steps.append(topk(div_pick_b, div_qfa[0], div_pick_q, div_qfa[1], -3.0))
        b_ids.append(div_pick_b)
        q_ids.append(div_pick_q)
    b = {"token_ids": b_ids, "text": "", "steps": b_steps}
    q = {"token_ids": q_ids, "text": "", "steps": q_steps}
    prefix = 0
    for x, y in zip(b_ids, q_ids):
        if x != y:
            break
        prefix += 1
    print(f"== {name}  greedy_prefix={prefix}/{min(len(b_ids), len(q_ids))}")
    stats = s.report_numerics(b, q, prefix)
    gate = s.OVERLAP_GATE * s.TOP_LOGPROBS
    ok = stats["steps"] > 0 and stats["max_delta"] <= s.DELTA_GATE and stats["mean_overlap"] >= gate
    print(f"  -> {'GREEN' if ok else 'RED'}\n")
    return ok


# prompt 0: 47 agreeing steps, max delta 0.0926, then ' Netherlands' vs ' United'
p0 = run(
    "prompt 0 (real: 47 steps, max delta 0.0926, qfa gap 0.0000)",
    list(range(100, 147)),
    [0.0926 if i == 12 else 0.012 for i in range(47)],
    div_base=(-0.5410, -0.9160),
    div_qfa=(-0.7128, -0.7128),
    div_pick_b=24844,
    div_pick_q=3516,
)
# prompt 1: 34 steps, max delta 0.2653, never diverges
p1 = run(
    "prompt 1 (real: 34 steps, max delta 0.2653, no divergence)",
    list(range(200, 234)),
    [0.2653 if i == 20 else 0.031 for i in range(34)],
    div_base=None,
    div_qfa=None,
    div_pick_b=None,
    div_pick_q=None,
)
# prompt 2: 2 steps, max delta 0.0807, then the symmetric ' is' / ' are' flip
p2 = run(
    "prompt 2 (real: 2 steps, max delta 0.0807, symmetric 0.2500 flip)",
    [1001, 1002],
    [0.0807, 0.006],
    div_base=(-0.5770, -0.8270),
    div_qfa=(-0.8267, -0.5767),
    div_pick_b=369,
    div_pick_q=513,
)

# A flat prompt: every step a near-tie, the also-rans reshuffling underneath a
# winner both runs agree on. Overlap alone would fail this - it is what put the
# real prompt 0 at 8.08 against a gate of 8.0 while all 64 chosen tokens
# matched - so the statistic has to abstain where no step is decisive.
flat = run(
    "flat prompt: near-ties throughout, top-k churns, every chosen token agrees",
    list(range(500, 530)),
    [0.05] * 30,
    div_base=None,
    div_qfa=None,
    div_pick_b=None,
    div_pick_q=None,
    gate=0.5,
    rival_lp=-0.30,
    q_shift=4,
)

# The MTP run measured on 2026-08-27. The baseline's pick moved only 0.23 -
# the number a one-sided delta would report - while the candidate that
# displaced it moved 0.96, and the output degenerated into a repetition loop.
# This has to come out RED, both because the run is genuinely bad and because
# it is what stops a one-sided delta from being reinstated.
mtp = run(
    "measured MTP: '\\n' moved 0.23 but the '\\n\\n' that won moved 0.96",
    [600, 601],
    [0.05, 0.05],
    div_base=(-1.4281, -1.9906),
    div_qfa=(-1.6576, -1.0326),
    div_pick_b=198,
    div_pick_q=271,
    gate=0.5,
)

# The same step once the verify write stops letting unconfirmed tokens set the
# V window's scale. The shift is scaled by the 3x the policy simulation gives
# (sim_qfa_spec_scale_policies.py), so this is a PREDICTION, not a measurement:
# it says the gate can tell the fixed run from the broken one above, not that
# the fix lands here. Replace it with the real numbers once the run exists.
mtp_fixed = run(
    "predicted MTP after the two-write fix: the same step at a third of the drift",
    [600, 601],
    [0.02, 0.02],
    div_base=(-1.4281, -1.9906),
    div_qfa=(-1.5046, -1.7823),
    div_pick_b=198,
    div_pick_q=198,
    gate=0.5,
)

bad = run(
    "sanity: broken cache, delta 1.2",
    list(range(300, 310)),
    [1.2] * 10,
    div_base=(-0.5, -0.9),
    div_qfa=(-2.0, -0.6),
    div_pick_b=400,
    div_pick_q=401,
    gate=0.5,
)

print(
    f"RESULT green={[p0, p1, p2]} flat_prompt_ok={flat} "
    f"mtp_rejected={not mtp} mtp_fixed_accepted={mtp_fixed} broken_rejected={not bad}"
)
sys.exit(0 if (p0 and p1 and p2 and flat and not mtp and not bad) else 1)
