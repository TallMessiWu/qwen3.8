#!/usr/bin/env python3
"""Find the FIRST op whose statistics differ between two msprobe dump trees.

Why this and not a full msprobe compare: the question here is not "how much do
these two runs drift" but "where do they stop being the same run". Two arms of
a single-variable A/B are supposed to be identical until the defect executes,
so the answer is the first op in execution order whose numbers move, and
everything after it is downstream noise. A full comparison buries that one line
under thousands.

Pair it with msprobe_survey.py, which decides whether two trees are comparable
at all; this script assumes they already are.

Usage:
    python3 scripts/debug/msprobe_first_divergence.py A_root B_root
    python3 scripts/debug/msprobe_first_divergence.py --step-a 3 --step-b 5 A B
    python3 scripts/debug/msprobe_first_divergence.py --rank rank0 --show 10 A B

Exit code: 0 when a divergence was located or the trees are identical,
2 when the trees could not be read or paired.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

STEP_RE = re.compile(r"^step(\d+)$")
# Statistics-mode fields msprobe writes per tensor. Shape and dtype are
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
    if step is None:
        # Default to the first step that actually recorded ops. profile_run and
        # the capture warmups burn step numbers without writing anything.
        for candidate in steps:
            path = candidate / rank / "dump.json"
            if path.exists():
                with path.open(encoding="utf-8") as fh:
                    payload = json.load(fh)
                if payload.get("data"):
                    return int(STEP_RE.match(candidate.name).group(1)), payload  # type: ignore[union-attr]
        raise SystemExit(f"no non-empty {rank}/dump.json under {root}")
    path = root / f"step{step}" / rank / "dump.json"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    with path.open(encoding="utf-8") as fh:
        return step, json.load(fh)


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


def differs(a, b, rtol: float) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, bool) or isinstance(b, bool):
            return a != b
        if math.isnan(a) and math.isnan(b):
            return False
        if math.isinf(a) or math.isinf(b):
            return a != b
        return abs(a - b) > rtol * max(1.0, abs(a), abs(b))
    return a != b


def compare(data_a: dict, data_b: dict, rtol: float, show: int) -> int:
    keys_a, keys_b = list(data_a), list(data_b)
    print(f"ops: A={len(keys_a)}  B={len(keys_b)}")

    if keys_a != keys_b:
        # Op sequences that differ in shape are themselves the finding: one arm
        # executed something the other did not.
        for index, (ka, kb) in enumerate(zip(keys_a, keys_b)):
            if ka != kb:
                print(f"\nop SEQUENCES diverge at index {index}:\n  A: {ka}\n  B: {kb}")
                return 0
        longer, name = (keys_a, "A") if len(keys_a) > len(keys_b) else (keys_b, "B")
        extra = longer[min(len(keys_a), len(keys_b)) :]
        print(f"\nop sequences share a prefix; {name} has {len(extra)} extra ops, first: {extra[0]}")
        return 0

    reported = 0
    for key in keys_a:
        ea, eb = data_a[key], data_b[key]
        if not isinstance(ea, dict) or not isinstance(eb, dict):
            continue
        sa, sb = tensor_stats(ea), tensor_stats(eb)
        if len(sa) != len(sb):
            print(f"\n[{reported}] {key}\n  tensor count differs: A={len(sa)} B={len(sb)}")
            reported += 1
            if reported >= show:
                return 0
            continue
        for (label, ta), (_, tb) in zip(sa, sb):
            bad = [k for k in STAT_KEYS if k in ta and k in tb and differs(ta[k], tb[k], rtol)]
            if bad:
                print(f"\n[{reported}] {key}  ({label})")
                for field in bad:
                    print(f"      {field:<6} A={ta[field]!r:<24} B={tb[field]!r}")
                reported += 1
                break
        if reported >= show:
            return 0

    if reported == 0:
        print("\nno divergence: every op matched within tolerance")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trees", nargs=2, metavar=("A_ROOT", "B_ROOT"))
    ap.add_argument("--rank", default="rank0")
    ap.add_argument("--step-a", type=int, default=None)
    ap.add_argument("--step-b", type=int, default=None)
    ap.add_argument("--rtol", type=float, default=1e-6, help="relative tolerance for the statistics")
    ap.add_argument("--show", type=int, default=5, help="how many divergences to print before stopping")
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
    return compare(payload_a.get("data") or {}, payload_b.get("data") or {}, args.rtol, args.show)


if __name__ == "__main__":
    sys.exit(main())
