#!/usr/bin/env python3
"""扫描「值在错误的时刻被固定」这一族的代码形状。

ACL 图捕获只记录不执行,编译区里的 Python 求值会被烘死成捕获期的值。所以下面这些
写法只要第一次执行落在捕获期,就会永久留下未初始化的缓冲或过期的判据。已踩过两次:

- MoE `_fused_output_is_reduced` 在编译区求值一次 → 长 prompt 只吐一个 EOS
- C8 MXFP 的 `filled_caches.add` 在捕获期真跑了 → V 反量化为零 → attention 恰好吐零

这是**纯文本匹配**,必然有误报:它只负责把候选摆到眼前,判断仍然靠人。命中行所在
函数体内出现 `capturing` / `_EXTRA_CTX` 的标 GUARDED(多半已核对过,默认不列),
其余标 REVIEW。守卫判定按函数体而不是按行窗口——按窗口会被邻近函数里的守卫骗过去。

用法:
    python scripts/checks/scan_capture_timing.py [PATH ...]      # 默认扫 cwd 下的 vllm_ascend/
    python scripts/checks/scan_capture_timing.py --noisy PATH    # 加扫前向分配与 D2H 同步
    python scripts/checks/scan_capture_timing.py --all PATH      # 连 GUARDED 一起列
    python scripts/checks/scan_capture_timing.py --strict PATH   # 有 REVIEW 就退出 1
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# (名字, 正则, 是否属于噪音档, 为什么危险 / 该核对什么)
PATTERNS: list[tuple[str, re.Pattern[str], bool, str]] = [
    (
        "lazy-init-guard",
        re.compile(r"\bhasattr\s*\(\s*\w+\s*,\s*[\"']_\w+[\"']\s*\)"),
        False,
        "惰性初始化守卫。核对:守卫体里有没有分配/拷贝(.to(device) / .contiguous() / copy_ / 算术)?"
        "第一次执行可能落在图捕获期吗?draft 尤其危险——target 有 _warmup_and_capture 的 eager"
        " warmup 兜底,draft 没有,它第一次走到这段就是捕获本身。",
    ),
    (
        "done-flag-assign",
        re.compile(r"^\s*(?:\w+\.)+_?\w*(?:prepared|filled|initialized|inited|loaded|done|ready)\s*=\s*True"),
        False,
        "「已做」标记赋值。核对:被它守住的是设备侧操作吗?若是,标记必须包进 "
        "`if not _EXTRA_CTX.capturing:`,否则捕获期只记录不执行、标记却真置上了,缓冲永远是零。",
    ),
    (
        "identity-set-marker",
        re.compile(r"^\s*\w*(?:cache|caches|filled|seen|done|visited)\w*\.add\s*\("),
        False,
        "用集合按张量身份记「填过了」。同上:捕获期这行 Python 真跑,被它守着的 copy_ 没跑。",
    ),
    (
        "bool-snapshot",
        re.compile(r"^\s*\w+\s*=\s*self\.\w*(?:is_|_is_|enable|should|needs?_)\w*\s*$"),
        False,
        "把属性 / property 快照成 Python bool 再传下去。核对:这个值随运行时状态变吗"
        "(通信方式、容量、本步 token 数)?在编译区里它会被烘死成捕获期 dummy run 的值。"
        "出口是 custom op(dynamo 黑盒),不是 torch._dynamo.disable——vLLM 以 fullgraph=True 编译,"
        "那里 graph break 是报错不是降级。改的时候把**所有**消费点一起改:上游契约常是成对的,"
        "只改一处会从吐 EOS 变成整段乱码。",
    ),
    (
        "wait-stream",
        re.compile(r"\.wait_stream\s*\("),
        False,
        "整流等待。核对:图已入队并停在事件上时 wait_stream 会死锁;正解是在需要的那一点 "
        "record_event() 配 wait_event(),只等那一点。",
    ),
    (
        "alloc-in-forward",
        re.compile(r"^\s*(?!#).*\btorch\.(?:zeros|empty|ones|full|arange)\s*\("),
        True,
        "前向里现场分配。核对:若这段会被捕获,分配与清零本身会被录成图节点、每次 replay 重跑;"
        "跨 replay 复用的 buffer 必须在捕获外建好。",
    ),
    (
        "d2h-in-hot-path",
        re.compile(r"\.(?:item|tolist)\s*\(\s*\)|\.cpu\s*\(\s*\)|\.numpy\s*\(\s*\)"),
        True,
        "设备→主机同步。核对:热路径里会卡住 AsyncScheduler;捕获期非法;曾经意外串行化前几个 "
        "decode 步、掩盖住 AICPU 写 plan 与刷缓冲区的竞态。",
    ),
]

GUARD_HINT = re.compile(r"capturing|is_capture|graph_capture|_EXTRA_CTX")
DEF_LINE = re.compile(r"^(\s*)(?:async\s+)?def\s+\w+")


def function_spans(lines: list[str]) -> list[tuple[int, int]]:
    """把文件切成 [函数起始行, 结束行) 的区间(0-based,含起始行)。

    结束点取下一个缩进不深于本函数 def 的 def/class 行,够用且不需要真正解析。
    """
    starts: list[tuple[int, int]] = []  # (行号, def 的缩进)
    for idx, line in enumerate(lines):
        m = DEF_LINE.match(line)
        if m:
            starts.append((idx, len(m.group(1))))

    spans: list[tuple[int, int]] = []
    for pos, (start, indent) in enumerate(starts):
        end = len(lines)
        for nxt_start, nxt_indent in starts[pos + 1 :]:
            if nxt_indent <= indent:
                end = nxt_start
                break
        spans.append((start, end))
    return spans


def enclosing_span(spans: list[tuple[int, int]], idx: int) -> tuple[int, int]:
    """取包住 idx 的最内层函数区间;不在任何函数里就退回单行。"""
    best = (idx, idx + 1)
    for start, end in spans:
        if start <= idx < end and (end - start) <= (best[1] - best[0]) or (start <= idx < end and best == (idx, idx + 1)):
            best = (start, end)
    return best


def scan_file(path: Path, noisy: bool) -> list[tuple[int, str, str, int | None]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:  # 软失败:跳过读不了的文件,别中断整次扫描
        print(f"!! 跳过 {path}: {exc}", file=sys.stderr)
        return []

    spans = function_spans(lines)
    hits: list[tuple[int, str, str, int | None]] = []
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        for name, pattern, is_noisy, _ in PATTERNS:
            if is_noisy and not noisy:
                continue
            if not pattern.search(line):
                continue
            start, end = enclosing_span(spans, idx)
            guard_at = next((i + 1 for i in range(start, end) if GUARD_HINT.search(lines[i])), None)
            hits.append((idx + 1, name, line.strip(), guard_at))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="要扫的目录或文件,默认是 cwd 下的 vllm_ascend/")
    parser.add_argument("--noisy", action="store_true", help="加扫前向分配与 D2H 同步(误报很多)")
    parser.add_argument("--all", action="store_true", help="连 GUARDED 的一起列出")
    parser.add_argument("--strict", action="store_true", help="存在 REVIEW 候选时退出码 1")
    args = parser.parse_args()

    roots = [Path(p) for p in args.paths] or [Path("vllm_ascend")]
    missing = [p for p in roots if not p.exists()]
    if missing:
        print(f"RED  路径不存在: {', '.join(str(p) for p in missing)}", file=sys.stderr)
        print("     在某个 vllm-ascend worktree 里跑,或显式传路径。", file=sys.stderr)
        return 2

    files: list[Path] = []
    for root in roots:
        files.extend(sorted(root.rglob("*.py")) if root.is_dir() else [root])

    review = guarded_total = 0
    seen_shapes: set[str] = set()
    for path in files:
        hits = scan_file(path, args.noisy)
        shown = [h for h in hits if args.all or h[3] is None]
        guarded_total += sum(1 for h in hits if h[3] is not None)
        if not shown:
            continue
        print(f"\n=== {path} ===")
        for lineno, name, text, guard_at in shown:
            if guard_at is None:
                review += 1
                seen_shapes.add(name)
                print(f"  REVIEW  {lineno:>5}  [{name}] {text[:110]}")
            else:
                print(f"  GUARDED {lineno:>5}  [{name}] {text[:110]}  (守卫见 :{guard_at})")

    if seen_shapes:
        print("\n--- 命中的形状要核对什么 ---")
        for name, _, _, note in PATTERNS:
            if name in seen_shapes:
                print(f"[{name}] {note}\n")

    print(f"候选 REVIEW={review} GUARDED={guarded_total}(同一函数体内出现过 capturing / _EXTRA_CTX)")
    print("纯文本匹配,误报是预期内的;REVIEW 不等于 bug,只等于「没人核对过」。")
    if review and args.strict:
        print("RED  --strict 下存在未核对候选")
        return 1
    print("GREEN 扫描完成" if not review else "REVIEW 扫描完成,逐条核对上面的候选")
    return 0


if __name__ == "__main__":
    sys.exit(main())
