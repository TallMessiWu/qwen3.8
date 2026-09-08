#!/usr/bin/env python3
"""Survey msprobe dump trees so eager and graph steps can be aligned by content.

The step index is NOT comparable between an eager dump and a graph dump: in
graph mode model_runner_v1 starts the debugger at the end of load_model (only
when cudagraph_mode != NONE) and every _dummy_run then calls
_finalize_dump_data(dump=False), which still advances debugger.step(). So
profile_run and each capture warmup consume step numbers that the eager run
never spends. Align by what a step *contains* instead -- the token count of the
forward -- which is what this prints.

Usage:
    python3 scripts/debug/msprobe_survey.py eager graph

RED/GREEN judgement is printed at the end:
  - GREEN: both trees contain a prefill-sized step (leading dim >> decode
    width) and the attention op is visible in both -> a compare is meaningful.
  - RED: the graph tree has no attention op or no prefill-sized step -> the
    dump cannot answer a QFA/attention question and re-collection is needed.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

# Substrings that identify the C8_MXFP attention operator in a dump key.
ATTN_MARKERS = ("quant_flash", "fused_infer_attention", "npu_dynamic_mx_quant", "reshape_and_cache")

STEP_RE = re.compile(r"^step(\d+)$")
RANK_RE = re.compile(r"^rank(\d+)?$")


def leading_dims(entry: dict) -> list[int]:
    """Leading dimension of every tensor in an entry's inputs."""
    dims = []
    for item in entry.get("input_args") or []:
        if isinstance(item, dict) and item.get("type") == "torch.Tensor":
            shape = item.get("shape") or []
            if shape:
                dims.append(shape[0])
    return dims


def survey_rank(dump_json: Path) -> dict:
    try:
        with dump_json.open(encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:  # noqa: BLE001 - a bad file must not kill the sweep
        return {"error": f"{type(exc).__name__}: {exc}"}

    data = payload.get("data") or {}
    dims: Counter = Counter()
    attn_hits: Counter = Counter()
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        dims.update(leading_dims(entry))
        for marker in ATTN_MARKERS:
            if marker in key:
                attn_hits[marker] += 1
    return {
        "task": payload.get("task"),
        "level": payload.get("level"),
        "entries": len(data),
        "dims": dims,
        "attn": attn_hits,
        "first_keys": list(data)[:3],
    }


def survey_tree(root: Path) -> list[dict]:
    rows = []
    steps = sorted(
        (p for p in root.iterdir() if p.is_dir() and STEP_RE.match(p.name)),
        key=lambda p: int(STEP_RE.match(p.name).group(1)),  # type: ignore[union-attr]
    )
    for step in steps:
        ranks = sorted(p for p in step.iterdir() if p.is_dir() and RANK_RE.match(p.name))
        for rank in ranks:
            dump_json = rank / "dump.json"
            row = {
                "step": int(STEP_RE.match(step.name).group(1)),  # type: ignore[union-attr]
                "rank": rank.name,
                "construct": (rank / "construct.json").exists(),
                "stack": (rank / "stack.json").exists(),
            }
            row.update(survey_rank(dump_json) if dump_json.exists() else {"error": "no dump.json"})
            rows.append(row)
    return rows


def print_tree(root: Path, rows: list[dict]) -> None:
    print(f"\n===== {root} =====")
    print(f"{'step':>5} {'rank':>6} {'cons':>5} {'entries':>8}  {'top leading dims (count)':<40} attn ops")
    for row in rows:
        if "error" in row:
            print(f"{row['step']:>5} {row['rank']:>6} {'-':>5} {'-':>8}  ERROR: {row['error']}")
            continue
        top = ", ".join(f"{d}x{n}" for d, n in row["dims"].most_common(3)) or "-"
        attn = ", ".join(f"{k}:{v}" for k, v in row["attn"].items()) or "NONE"
        print(f"{row['step']:>5} {row['rank']:>6} {str(row['construct']):>5} {row['entries']:>8}  {top:<40} {attn}")


def dominant_dim(row: dict) -> int:
    if "dims" not in row or not row["dims"]:
        return 0
    return row["dims"].most_common(1)[0][0]


def verdict(name: str, rows: list[dict]) -> bool:
    ok = True
    rank0 = [r for r in rows if r.get("rank") in ("rank0", "rank") and "dims" in r]
    if not rank0:
        print(f"[RED ] {name}: no readable rank0 dump.json")
        return False

    widths = sorted({dominant_dim(r) for r in rank0})
    print(f"[info] {name}: rank0 dominant token widths across steps = {widths}")
    if max(widths) < 32:
        print(f"[RED ] {name}: no prefill-sized step (max width {max(widths)}); the long prompt is not in this dump")
        ok = False
    else:
        prefill_steps = [r["step"] for r in rank0 if dominant_dim(r) == max(widths)]
        print(f"[GREEN] {name}: prefill-sized step(s) = {prefill_steps} (width {max(widths)})")

    with_attn = [r["step"] for r in rank0 if r.get("attn")]
    if not with_attn:
        print(f"[RED ] {name}: no attention op in any step -- the dump cannot answer a QFA question")
        ok = False
    else:
        print(f"[GREEN] {name}: attention op present in steps {with_attn[:5]}{'...' if len(with_attn) > 5 else ''}")
    return ok


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    all_ok = True
    surveys = []
    for arg in argv[1:]:
        root = Path(arg)
        if not root.is_dir():
            print(f"[RED ] {root} is not a directory")
            all_ok = False
            continue
        rows = survey_tree(root)
        surveys.append((root, rows))
        print_tree(root, rows)

    print("\n===== verdict =====")
    for root, rows in surveys:
        all_ok &= verdict(root.name, rows)

    if len(surveys) == 2:
        (_, a), (_, b) = surveys
        a0 = [r for r in a if r.get("rank") == "rank0" and "dims" in r]
        b0 = [r for r in b if r.get("rank") == "rank0" and "dims" in r]
        print(f"\n[info] step counts: {surveys[0][0].name}={len(a0)} {surveys[1][0].name}={len(b0)}")
        print("[info] align by the token-width column above, NOT by the step index.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
