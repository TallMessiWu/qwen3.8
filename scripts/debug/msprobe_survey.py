#!/usr/bin/env python3
"""Survey msprobe dump trees before spending any time comparing them.

Two things make a naive eager-vs-graph comparison worthless, and this script
checks for both.

1. The step index is NOT comparable between the two trees. model_runner_v1
   picks the dumper off cudagraph_mode -- PrecisionDebugger when it is NONE,
   AclGraphDumper otherwise -- and only starts the graph one at the end of
   load_model. Every _dummy_run then calls _finalize_dump_data(dump=False),
   which advances debugger.step() without writing anything, so profile_run and
   each capture warmup burn a step number that an eager run never spends.
   Align by what a step *contains* -- the token width of its forward -- which
   is the column this prints.

2. A dump only answers a question about a bug if the bug happened while it was
   being collected. The truncation symptom is "one completion token", i.e. a
   prefill step followed by exactly one decode step. A tree with many decode
   steps recorded a healthy generation, and comparing two healthy runs only
   measures ordinary graph-vs-eager drift.

Usage:
    python3 scripts/debug/msprobe_survey.py eager graph
    python3 scripts/debug/msprobe_survey.py --mtp 3 eager graph

RED/GREEN judgement is printed at the end. Every RED means "do not compare
these trees yet", with the reason named.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Substrings identifying interesting ops in a dump key. GDN matters because the
# graph-vs-eager divergence was last bisected to the GDN layers; attention
# matters because QFA is the op under test.
MARKERS = {
    "attn": ("quant_flash", "fused_infer_attention", "npu_dynamic_mx_quant", "reshape_and_cache"),
    "gdn": ("gated_delta", "chunk_gated", "causal_conv", "conv1d", "recurrent"),
    "moe": ("moe_", "grouped_matmul", "moe_init_routing", "all_gather", "dispatch"),
}

STEP_RE = re.compile(r"^step(\d+)$")
RANK_RE = re.compile(r"^rank(\d+)?$")


def leading_dims(entry: dict) -> list[int]:
    """Leading dimension of every input tensor in one dump entry."""
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
    hits: Counter = Counter()
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        dims.update(leading_dims(entry))
        lowered = key.lower()
        for group, markers in MARKERS.items():
            if any(m in lowered for m in markers):
                hits[group] += 1
    return {
        "task": payload.get("task"),
        "level": payload.get("level"),
        "entries": len(data),
        "dims": dims,
        "hits": hits,
    }


def survey_tree(root: Path) -> list[dict]:
    rows = []
    steps = sorted(
        (p for p in root.iterdir() if p.is_dir() and STEP_RE.match(p.name)),
        key=lambda p: int(STEP_RE.match(p.name).group(1)),  # type: ignore[union-attr]
    )
    for step in steps:
        for rank in sorted(p for p in step.iterdir() if p.is_dir() and RANK_RE.match(p.name)):
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
    header = f"{'step':>5} {'rank':>6} {'cons':>5} {'stck':>5} {'entries':>8}  {'top leading dims':<32} markers"
    print(header)
    for row in rows:
        if "error" in row:
            print(f"{row['step']:>5} {row['rank']:>6} {'-':>5} {'-':>5} {'-':>8}  ERROR: {row['error']}")
            continue
        top = ", ".join(f"{d}x{n}" for d, n in row["dims"].most_common(3)) or "-"
        marks = ", ".join(f"{k}:{v}" for k, v in sorted(row["hits"].items())) or "NONE"
        print(
            f"{row['step']:>5} {row['rank']:>6} {str(row['construct']):>5} {str(row['stack']):>5} "
            f"{row['entries']:>8}  {top:<32} {marks}"
        )


def dominant_dim(row: dict) -> int:
    if not row.get("dims"):
        return 0
    return row["dims"].most_common(1)[0][0]


def verdict(name: str, rows: list[dict], decode_width: int) -> bool:
    ok = True
    rank0 = [r for r in rows if r.get("rank") == "rank0" and "dims" in r]
    if not rank0:
        print(f"[RED ] {name}: no readable rank0 dump.json")
        return False

    widths = {r["step"]: dominant_dim(r) for r in rank0}
    print(f"[info] {name}: {len(rank0)} rank0 steps, dominant widths = {sorted(set(widths.values()))}")

    # A prefill step is one materially wider than a uniform decode step.
    prefill = [s for s, w in widths.items() if w > decode_width * 4]
    if not prefill:
        print(f"[RED ] {name}: no prefill-sized step (max width {max(widths.values())}); long prompt not in this dump")
        ok = False
    else:
        print(f"[GREEN] {name}: prefill step(s) {prefill} at width {max(widths.values())}")

    # The symptom under investigation is a single completion token. One decode
    # step means it reproduced; many decode steps mean this dump recorded a
    # healthy generation and has nothing to say about the bug.
    decode = [s for s, w in widths.items() if 0 < w <= decode_width]
    empty = [s for s, w in widths.items() if w == 0]
    print(f"[info] {name}: {len(decode)} decode-shaped steps (width<={decode_width}), {len(empty)} empty steps")
    if len(decode) > 2:
        print(
            f"[RED ] {name}: {len(decode)} decode steps -- this run generated many tokens, so the "
            "one-token truncation did NOT reproduce here. Comparing it measures ordinary drift, not the bug."
        )
        ok = False
    elif decode:
        print(f"[GREEN] {name}: {len(decode)} decode step(s) -- consistent with the one-token truncation")

    # graph_visualize needs the full trio; statistics-only steps cannot be walked.
    incomplete = sorted({r["step"] for r in rows if "error" not in r and not (r["construct"] and r["stack"])})
    if incomplete:
        print(f"[warn] {name}: steps without construct.json+stack.json (graph_visualize cannot use them): {incomplete}")

    for group in MARKERS:
        seen = [r["step"] for r in rank0 if r.get("hits", {}).get(group)]
        if seen:
            print(f"[GREEN] {name}: {group} ops present in steps {seen[:6]}{'...' if len(seen) > 6 else ''}")
        else:
            print(f"[warn] {name}: no {group} op matched any dump key -- that layer is invisible in this tree")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trees", nargs="+", help="msprobe dump roots, e.g. eager graph")
    ap.add_argument(
        "--mtp",
        type=int,
        default=3,
        help="num_speculative_tokens; a uniform decode step is mtp+1 tokens wide per request (default 3)",
    )
    args = ap.parse_args()
    decode_width = args.mtp + 1

    all_ok = True
    surveys = []
    for arg in args.trees:
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
        all_ok &= verdict(root.name, rows, decode_width)
        print()

    if len(surveys) == 2:
        (ra, a), (rb, b) = surveys
        a0 = [r for r in a if r.get("rank") == "rank0" and "dims" in r]
        b0 = [r for r in b if r.get("rank") == "rank0" and "dims" in r]
        print(f"[info] step counts: {ra.name}={len(a0)} {rb.name}={len(b0)}")
        wa = {dominant_dim(r) for r in a0}
        wb = {dominant_dim(r) for r in b0}
        shared = sorted(wa & wb)
        print(f"[info] token widths present in both: {shared or 'NONE'}")
        print("[info] pair steps by that width, never by step index.")
        if not shared:
            print("[RED ] the two trees share no token width -- they did not run the same request")
            all_ok = False

    print(f"\n{'[GREEN] survey passed' if all_ok else '[RED ] survey failed -- see reasons above'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
