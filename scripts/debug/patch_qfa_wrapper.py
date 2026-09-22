#!/usr/bin/env python3
"""In-place patch the installed QFA wrapper so prefill sizes metadata by num_heads_q.

Applies the fix from TallMessiWu/ops-transformer@fix-qfa-metadata-prefill-capacity
to the copy already installed under CANN, so it can be verified without building
anything -- the wrapper is pure Python, the .py in site-packages IS what runs.

Only the three lines that matter are touched, not the whole file: the installed
package is a release build and the repo is master, so copying the file over would
drag in unrelated differences and make the comparison dirty.

Reversible on purpose. The whole point is to flip the same machine back and
forth:

    python3 scripts/debug/patch_qfa_wrapper.py --status
    python3 scripts/debug/patch_qfa_wrapper.py            # apply
    python3 scripts/checks/qfa_metadata_capacity.py       # expect exit 0 now
    python3 scripts/debug/patch_qfa_wrapper.py --revert
    python3 scripts/checks/qfa_metadata_capacity.py       # expect exit 1 again

A .bak of the original sits next to the file; --revert restores from it.

Delete this script once the fix lands upstream.
"""

from __future__ import annotations

import argparse
import inspect
import shutil
import sys
from pathlib import Path

MARKER = "head_num = num_heads_kv"

OLD_SIG = "def _calculate_max_schedule_size(batch_size, num_heads_kv):"
NEW_SIG = "def _calculate_max_schedule_size(batch_size, num_heads_kv, num_heads_q=None, layout_q_descale=None):"

OLD_BODY = """    aic_num, aiv_num = _get_core_nums()
    fa_size = aic_num * METADATA_STRIDE * batch_size * num_heads_kv
    fd_size = aiv_num * METADATA_STRIDE * batch_size * num_heads_kv"""
NEW_BODY = """    # AICPU: baseInfo.kvHeadNum = isDecode ? numHeadsKv : numHeadsQ, and
    # CalcGridInfoSection counts its inner loop to GetKvHeadNum(), so prefill's
    # worst-case sectionNum is batch*num_heads_q. layout None -> take the larger.
    head_num = num_heads_kv
    if num_heads_q is not None and layout_q_descale != "N2TGD":
        head_num = num_heads_q
    aic_num, aiv_num = _get_core_nums()
    fa_size = aic_num * METADATA_STRIDE * batch_size * head_num
    fd_size = aiv_num * METADATA_STRIDE * batch_size * head_num"""

OLD_META_CALL = "            max_schedule_size = _calculate_max_schedule_size(b_size, num_heads_kv)"
NEW_META_CALL = "            max_schedule_size = _calculate_max_schedule_size(b_size, num_heads_kv, num_heads_q, layout_q_descale)"

OLD_REAL_CALL = "    max_schedule_size = _calculate_max_schedule_size(batch_size, num_heads_kv)"
NEW_REAL_CALL = "    max_schedule_size = _calculate_max_schedule_size(batch_size, num_heads_kv, num_heads_q, layout_q_descale)"

EDITS = [
    ("signature", OLD_SIG, NEW_SIG),
    ("head count", OLD_BODY, NEW_BODY),
    ("meta-kernel call site", OLD_META_CALL, NEW_META_CALL),
    ("device call site", OLD_REAL_CALL, NEW_REAL_CALL),
]


def locate() -> Path:
    """Ask the installed package where it lives rather than hardcoding a path."""
    from cann_ops_transformer.ops import quant_flash_attn_metadata as op

    return Path(inspect.getfile(op)).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--revert", action="store_true", help="restore the .bak")
    parser.add_argument("--status", action="store_true", help="report only, change nothing")
    parser.add_argument("--file", help="override the target path (default: ask the package)")
    args = parser.parse_args()

    try:
        target = Path(args.file).resolve() if args.file else locate()
    except ImportError as exc:
        print(f"[RED] cannot import cann_ops_transformer: {exc}")
        return 1
    backup = target.with_suffix(target.suffix + ".bak")
    print(f"target: {target}")
    print(f"backup: {backup} ({'exists' if backup.exists() else 'not yet'})")

    text = target.read_text(encoding="utf-8")
    patched = MARKER in text
    print(f"state : {'PATCHED' if patched else 'original'}")

    if args.status:
        return 0

    if args.revert:
        if not backup.exists():
            print("[RED] no .bak to restore from")
            return 1
        shutil.copy2(backup, target)
        print("[GREEN] reverted to the original wrapper")
        return 0

    if patched:
        print("[GREEN] already patched, nothing to do")
        return 0

    for label, old, _ in EDITS:
        if old not in text:
            print(f"[RED] cannot find the '{label}' block -- this wrapper differs from")
            print("      the one this patch was written for. Nothing was changed.")
            print("      Expected to find:")
            print("        " + old.strip().splitlines()[0])
            return 1

    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"[info] saved {backup.name}")
    for label, old, new in EDITS:
        text = text.replace(old, new, 1)
        print(f"  patched: {label}")
    target.write_text(text, encoding="utf-8", newline="")

    # A syntax error here would take the whole serving stack down, so check it
    # before handing the file back, and roll back if it does not compile.
    import py_compile

    try:
        py_compile.compile(str(target), doraise=True)
    except py_compile.PyCompileError as exc:
        shutil.copy2(backup, target)
        print(f"[RED] patched file does not compile, rolled back: {exc}")
        return 1
    print("[GREEN] patched and compiles. Now rerun scripts/checks/qfa_metadata_capacity.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
