#!/usr/bin/env python3
"""Summarize the QFA length dump written by the debug-qfa-kv-lens probe.

The operator team asked what seqused_kv / cu_seqlens_kv hold on each QFA call.
This path never passes cu_seqlens_kv -- the KV side is block_table +
seqused_kv -- so the dump carries cu_seqlens_q and seqused_kv exactly as the
operator received them, graph padding included, one JSON line per call.

    python3 qfa_len_summary.py <dump dir | .jsonl ...> [--group-size G]
        [--since UNIX_TIME] [--export typical_calls.json]

Stdlib only, so it runs on the server or wherever the files are copied.

RED/GREEN judges only whether what the operator was handed is self-consistent:
the two length tensors agree on the batch, cu_seqlens_q ends at the token
count, boundaries never go backwards, block_table has a row per request.
Everything else is descriptive -- batch sizes, per-request Q_S, KV lengths, and
above all how unequal the KV lengths inside one call are.

Padding is identified by the probe from the raw buffer (zero-filled slots)
before the sanitize turns it into KV length 1, and it always sits at the tail.

Delete this together with the debug-qfa-kv-lens branch once answered.
"""

import argparse
import glob
import json
import os
import statistics
import sys
from collections import Counter

# The operator doc's boundary between the two q-scale layouts.
N2TGD_MAX_G_TIMES_QS = 80


def load(paths: list[str], since: float | None) -> list[dict]:
    files: list[str] = []
    for path in paths:
        files += sorted(glob.glob(os.path.join(path, "*.jsonl"))) if os.path.isdir(path) else [path]
    if not files:
        sys.exit(f"RED  no .jsonl files under {paths}")
    records = []
    for name in files:
        with open(name) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a server killed mid-write leaves one torn line
                if since is None or rec["time"] >= since:
                    rec["_file"] = os.path.basename(name)
                    records.append(rec)
    print(f"读入 {len(files)} 个文件，{len(records)} 条调用记录")
    return records


def pct(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    return sorted_values[min(len(sorted_values) - 1, int(q * (len(sorted_values) - 1) + 0.5))]


def describe(values: list[float]) -> str:
    if not values:
        return "n=0"
    s = sorted(values)
    return (
        f"n={len(s)}  min={s[0]:g}  p50={pct(s, 0.5):g}  p90={pct(s, 0.9):g}  "
        f"p99={pct(s, 0.99):g}  max={s[-1]:g}  mean={statistics.fmean(s):.4g}"
    )


def derive(rec: dict) -> dict:
    cu, kv = rec["cu_seqlens_q"], rec["seqused_kv"]
    batch = len(kv)
    real = max(0, batch - rec["num_padding_kv_slots"])
    q_lens = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
    max_seqlen_q = rec["max_query_len"] or rec["num_actual_tokens"]
    return {
        "batch": batch,
        "real": real,
        "real_kv": kv[:real],
        "real_q": q_lens[:real],
        "max_seqlen_q": max_seqlen_q,
    }


def consistency(records: list[dict]) -> bool:
    bad = Counter()
    for rec in records:
        cu, kv = rec["cu_seqlens_q"], rec["seqused_kv"]
        if len(cu) - 1 != len(kv):
            bad["len(cu_seqlens_q)-1 != len(seqused_kv)"] += 1
        if cu and cu[-1] != rec["num_actual_tokens"]:
            bad["cu_seqlens_q[-1] != num_actual_tokens (TND 约束)"] += 1
        if any(cu[i + 1] < cu[i] for i in range(len(cu) - 1)):
            bad["cu_seqlens_q 非单调"] += 1
        rows = rec.get("block_table_rows")
        if rows is not None and rows < len(kv):
            bad["block_table 行数 < batch"] += 1
    print("\n== 一致性（算子拿到的入参是否自洽）==")
    if not bad:
        print(f"GREEN  {len(records)} 条全部自洽")
        return True
    for what, n in bad.items():
        print(f"RED    {what}: {n} / {len(records)} 条")
    return False


def section(title: str, records: list[dict], group_size: int | None) -> list[tuple[dict, dict]]:
    print(f"\n== {title}：{len(records)} 条 ==")
    if not records:
        return []
    pairs = [(rec, derive(rec)) for rec in records]
    modes = Counter((r["graph_mode"], r["attn_state"]) for r in records)
    print("图模式 × attn_state：" + "，".join(f"{m}/{s}={n}" for (m, s), n in modes.most_common()))

    print("\n-- batch --")
    print("送进算子的 batch（含 padding）：", describe([d["batch"] for _, d in pairs]))
    print("其中真实请求数：            ", describe([d["real"] for _, d in pairs]))
    padded = sum(1 for _, d in pairs if d["batch"] > d["real"])
    print(f"带 padding 的调用：{padded} / {len(pairs)}")

    print("\n-- query --")
    print("每个真实请求的 Q_S：", describe([q for _, d in pairs for q in d["real_q"]]))
    seqlen_q = Counter(d["max_seqlen_q"] for _, d in pairs)
    print("max_seqlen_q 分布：" + "，".join(f"{k}:{v}" for k, v in sorted(seqlen_q.items())[:12]))
    no_mask = sum(1 for _, d in pairs if d["max_seqlen_q"] == 1)
    print(f"走 NO_MASK（max_seqlen_q==1）：{no_mask} / {len(pairs)}")
    if group_size:
        n2tgd = sum(1 for _, d in pairs if group_size * d["max_seqlen_q"] <= N2TGD_MAX_G_TIMES_QS)
        print(f"走 N2TGD decode 模板（G={group_size}，G*Q_S<={N2TGD_MAX_G_TIMES_QS}）：{n2tgd} / {len(pairs)}")

    print("\n-- seqused_kv（只算真实请求）--")
    print("单个请求的 KV 长度：", describe([k for _, d in pairs for k in d["real_kv"]]))
    print("单次调用内的最长 KV：", describe([max(d["real_kv"]) for _, d in pairs if d["real_kv"]]))
    multi = [d for _, d in pairs if d["real"] >= 2]
    if multi:
        equal = sum(1 for d in multi if min(d["real_kv"]) == max(d["real_kv"]))
        ratios = [max(d["real_kv"]) / min(d["real_kv"]) for d in multi]
        cvs = [statistics.pstdev(d["real_kv"]) / statistics.fmean(d["real_kv"]) for d in multi]
        print(f"\n同一次调用内 KV 是否等长（真实请求 >= 2 的 {len(multi)} 条）：")
        print(f"  完全等长：{equal} 条（{equal / len(multi):.1%}）")
        for bound in (1.1, 1.5, 2, 10):
            n = sum(1 for r in ratios if r > bound)
            print(f"  最长/最短 > {bound:<4}：{n} 条（{n / len(multi):.1%}）")
        print("  最长/最短：", describe(ratios))
        print("  变异系数 std/mean：", describe(cvs))
    return pairs


def pick_typical(pairs: list[tuple[dict, dict]]) -> dict:
    multi = [(r, d) for r, d in pairs if d["real"] >= 2]
    picks: dict[str, tuple[dict, dict]] = {}
    if pairs:
        picks["largest_batch"] = max(pairs, key=lambda p: p[1]["real"])
        picks["longest_kv"] = max(pairs, key=lambda p: max(p[1]["real_kv"] or [0]))
    if multi:
        picks["most_unequal_kv"] = max(multi, key=lambda p: max(p[1]["real_kv"]) / min(p[1]["real_kv"]))
        modal = Counter(d["real"] for _, d in multi).most_common(1)[0][0]
        at_modal = sorted(
            (p for p in multi if p[1]["real"] == modal),
            key=lambda p: statistics.pstdev(p[1]["real_kv"]) / statistics.fmean(p[1]["real_kv"]),
        )
        picks[f"median_spread_at_modal_batch_{modal}"] = at_modal[len(at_modal) // 2]
    keep = (
        "forward", "time", "graph_mode", "attn_state", "num_actual_tokens", "max_query_len",
        "num_padding_kv_slots", "cu_seqlens_q", "seqused_kv",
    )  # fmt: skip
    return {name: {k: rec[k] for k in keep} for name, (rec, _) in picks.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="dump 目录或 .jsonl 文件")
    parser.add_argument("--group-size", type=int, help="G = 每个 rank 的 num_heads / num_kv_heads，给了才统计 layout")
    parser.add_argument("--since", type=float, help="只统计此 Unix 时间之后的调用（切掉压测开始前的零星请求）")
    parser.add_argument("--export", help="把几条代表性调用的完整入参写成 JSON，给算子侧直接回放")
    args = parser.parse_args()

    records = load(args.paths, args.since)
    ok = consistency(records)
    target = section("target 模型", [r for r in records if not r["draft"]], args.group_size)
    draft = section("draft（MTP）模型", [r for r in records if r["draft"]], args.group_size)

    if args.export:
        typical = {"target": pick_typical(target), "draft": pick_typical(draft)}
        with open(args.export, "w") as fh:
            json.dump(typical, fh, indent=1)
        print(f"\n代表性调用已写到 {args.export}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
