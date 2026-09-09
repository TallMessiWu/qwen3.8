#!/usr/bin/env python3
"""Locate where two msprobe dump trees stop being the same run.

Why this and not `msprobe graph_visualize`: the question in a single-variable
A/B is not "how much do these two runs drift" but "where do they stop being the
same run". Both arms are supposed to be identical until the defect executes, so
the answer is the first op in execution order whose numbers move, and
everything after it is downstream noise. A full visual comparison buries that
one line under thousands of nodes.

Two things are reported, in this order, because the first often makes the
second unnecessary:

1. Ops whose statistics msprobe could not compute. That is what its "Invalid
   statistics detected. Please use tensor mode to collect the affected data"
   warning means: the task hit NaN/Inf or another numerical anomaly. An op that
   is invalid in ONE arm only is not an artefact of the dump -- it is the bug,
   already named.
2. The first op whose statistics differ between the arms.

Pair with msprobe_survey.py, which decides whether two trees are comparable at
all; this script assumes they already are.

Usage:
    python3 scripts/debug/msprobe_first_divergence.py A_ROOT B_ROOT
    python3 scripts/debug/msprobe_first_divergence.py --step-a 3 --step-b 5 A B
    python3 scripts/debug/msprobe_first_divergence.py --grep quant_flash A B

Exit code: 0 when the trees were read and compared, 2 when they were not.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

STEP_RE = re.compile(r"^step(\d+)$")
# Statistics-mode fields msprobe writes per tensor. shape and dtype are
# compared too: a changed shape is a divergence even when the numbers are
# absent, and it is usually the more legible symptom.
STAT_KEYS = ("Max", "Min", "Mean", "Norm", "shape", "dtype")


def load_step(root: Path, rank: str, step: int | None) -> tuple[int, dict]:
    steps = sorted(
        (p for p in root.iterdir() if p.is_dir() and STEP_RE.match(p.name)),
        key=lambda p: int(STEP_RE.match(p.name).group(1)),  # type: ignore[union-attr]
    )
    if not steps:
        raise SystemExit(f"no stepN directories under {root}")
    if step is not None:
        path = root / f"step{step}" / rank / "dump.json"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        with path.open(encoding="utf-8") as fh:
            return step, json.load(fh)
    # Default to the first step that recorded anything: profile_run and the
    # capture warmups burn step numbers without writing ops.
    for candidate in steps:
        path = candidate / rank / "dump.json"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("data"):
            return int(STEP_RE.match(candidate.name).group(1)), payload  # type: ignore[union-attr]
    raise SystemExit(f"no non-empty {rank}/dump.json under {root}")


def load_stacks(root: Path, rank: str, step: int) -> dict:
    """stack.json turns an op key into the source lines that called it.

    That is the difference between "op number 4172 diverged" and "this line of
    this file diverged", so load it whenever the level produced one.
    """
    path = root / f"step{step}" / rank / "stack.json"
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        # A missing or malformed stack.json costs source locations, nothing
        # more; the numeric comparison is what matters and it still runs.
        return {}


def stack_for(stacks: dict, key: str, limit: int = 4) -> list[str]:
    if not stacks:
        return []
    entry = stacks.get(key)
    if entry is None:
        # Some msprobe versions key stack.json by an id shared across ops.
        for value in stacks.values():
            if isinstance(value, dict) and key in (value.get("api_list") or []):
                entry = value.get("stack_info")
                break
    if isinstance(entry, dict):
        entry = entry.get("stack_info")
    if isinstance(entry, str):
        entry = [entry]
    if not isinstance(entry, list):
        return []
    frames = [str(f).strip() for f in entry if isinstance(f, str)]
    # Frames inside torch/dynamo say nothing about this repo; ours first.
    ours = [f for f in frames if "vllm_ascend" in f or "vllm/" in f]
    return (ours or frames)[:limit]


def tensor_stats(entry: dict) -> list[tuple[str, dict]]:
    """Every tensor-shaped leaf in one dump entry, labelled by its position."""
    found: list[tuple[str, dict]] = []

    def walk(node, label: str) -> None:
        if isinstance(node, dict):
            if node.get("type") == "torch.Tensor" or "Max" in node:
                found.append((label, node))
                return
            for key, value in node.items():
                walk(value, f"{label}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{label}[{index}]")

    for section in ("input_args", "input_kwargs", "output"):
        if section in entry:
            walk(entry[section], section)
    return found


def invalid_reason(stat: dict) -> str | None:
    """Why msprobe could not summarise this tensor, if it could not.

    This is the content behind its invalid-statistics warning. NaN and Inf are
    reported separately from "absent" because only the first two mean the model
    actually produced a bad number.
    """
    values = {k: stat.get(k) for k in ("Max", "Min", "Mean", "Norm")}
    present = {k: v for k, v in values.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if not present:
        return "no statistics"
    nans = [k for k, v in present.items() if math.isnan(v)]
    infs = [k for k, v in present.items() if math.isinf(v)]
    if nans:
        return "NaN in " + ",".join(nans)
    if infs:
        return "Inf in " + ",".join(infs)
    return None


def scan_invalid(data: dict, grep: str | None) -> list[tuple[str, str, str]]:
    out = []
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        if grep and grep not in key:
            continue
        for label, stat in tensor_stats(entry):
            reason = invalid_reason(stat)
            if reason:
                out.append((key, label, reason))
    return out


def report_invalid(name_a: str, name_b: str, inv_a: list, inv_b: list) -> None:
    keys_a = {k for k, _, _ in inv_a}
    keys_b = {k for k, _, _ in inv_b}
    only_a, only_b, both = keys_a - keys_b, keys_b - keys_a, keys_a & keys_b
    print(f"\ninvalid statistics: A={len(keys_a)} ops, B={len(keys_b)} ops, shared={len(both)}")
    for label, keys, inv in ((name_a, only_a, inv_a), (name_b, only_b, inv_b)):
        if not keys:
            continue
        print(f"  !! bad in {label} ONLY ({len(keys)} ops) -- an op that goes NaN/Inf in one arm is the finding:")
        shown = 0
        for key, where, reason in inv:
            if key in keys:
                print(f"       {key}  ({where})  {reason}")
                shown += 1
                if shown >= 8:
                    print(f"       ... {len(keys) - shown} more")
                    break
    if both and not only_a and not only_b:
        print("  (the same ops are unsummarisable in both arms: an artefact of the dump, not a difference)")


def differs(a, b, rtol: float) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, bool) or isinstance(b, bool):
            return a != b
        if math.isnan(a) and math.isnan(b):
            return False
        if math.isinf(a) or math.isinf(b):
            return a != b
        return abs(a - b) > rtol * max(1.0, abs(a), abs(b))
    return a != b


def compare(data_a: dict, data_b: dict, rtol: float, show: int, stacks: dict, grep: str | None) -> None:
    keys_a = [k for k in data_a if not grep or grep in k]
    keys_b = [k for k in data_b if not grep or grep in k]
    print(f"\nops compared: A={len(keys_a)}  B={len(keys_b)}")

    if keys_a != keys_b:
        # Op sequences that differ are themselves the finding: one arm executed
        # something the other did not.
        for index, (ka, kb) in enumerate(zip(keys_a, keys_b)):
            if ka != kb:
                print(f"\nop SEQUENCES diverge at index {index}:\n  A: {ka}\n  B: {kb}")
                for frame in stack_for(stacks, ka):
                    print(f"      at {frame}")
                return
        longer, name = (keys_a, "A") if len(keys_a) > len(keys_b) else (keys_b, "B")
        extra = longer[min(len(keys_a), len(keys_b)) :]
        print(f"\nop sequences share a prefix; {name} has {len(extra)} extra ops, first: {extra[0]}")
        return

    reported = skipped = 0
    for key in keys_a:
        ea, eb = data_a[key], data_b[key]
        if not isinstance(ea, dict) or not isinstance(eb, dict):
            continue
        sa, sb = tensor_stats(ea), tensor_stats(eb)
        if len(sa) != len(sb):
            print(f"\n[{reported}] {key}\n      tensor count differs: A={len(sa)} B={len(sb)}")
            reported += 1
            if reported >= show:
                return
            continue
        for (label, ta), (_, tb) in zip(sa, sb):
            if invalid_reason(ta) or invalid_reason(tb):
                skipped += 1
                continue
            bad = [k for k in STAT_KEYS if k in ta and k in tb and differs(ta[k], tb[k], rtol)]
            if bad:
                print(f"\n[{reported}] {key}  ({label})")
                for field in bad:
                    print(f"      {field:<6} A={ta[field]!r:<24} B={tb[field]!r}")
                for frame in stack_for(stacks, key):
                    print(f"      at {frame}")
                reported += 1
                break
        if reported >= show:
            return

    if skipped:
        print(f"\n{skipped} tensor(s) not compared: statistics unsummarisable in at least one arm")
    if reported == 0:
        print("\nno divergence: every comparable op matched within tolerance")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trees", nargs=2, metavar=("A_ROOT", "B_ROOT"))
    ap.add_argument("--rank", default="rank0")
    ap.add_argument("--step-a", type=int, default=None)
    ap.add_argument("--step-b", type=int, default=None)
    ap.add_argument("--rtol", type=float, default=1e-6, help="relative tolerance for the statistics")
    ap.add_argument("--show", type=int, default=5, help="how many divergences to print before stopping")
    ap.add_argument("--grep", default=None, help="only consider op keys containing this substring")
    args = ap.parse_args()

    root_a, root_b = Path(args.trees[0]), Path(args.trees[1])
    for root in (root_a, root_b):
        if not root.is_dir():
            print(f"not a directory: {root}", file=sys.stderr)
            return 2

    step_a, payload_a = load_step(root_a, args.rank, args.step_a)
    step_b, payload_b = load_step(root_b, args.rank, args.step_b)
    print(f"A = {root_a}/step{step_a}/{args.rank}  task={payload_a.get('task')} level={payload_a.get('level')}")
    print(f"B = {root_b}/step{step_b}/{args.rank}  task={payload_b.get('task')} level={payload_b.get('level')}")

    data_a = payload_a.get("data") or {}
    data_b = payload_b.get("data") or {}
    report_invalid(str(root_a.name), str(root_b.name), scan_invalid(data_a, args.grep), scan_invalid(data_b, args.grep))

    stacks = load_stacks(root_a, args.rank, step_a) or load_stacks(root_b, args.rank, step_b)
    compare(data_a, data_b, args.rtol, args.show, stacks, args.grep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
