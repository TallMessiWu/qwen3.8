#!/usr/bin/env python3
"""Check the delivered QFA operator contract against what attention_v1.py assumes.

vllm_ascend/attention/attention_v1.py calls cann_ops_transformer's
``quant_flash_attn`` / ``quant_flash_attn_metadata`` with a fixed keyword set
whose comments say "delivery signature (verified on-device)". Those operators
ship with the CANN toolkit, so swapping the CANN package can move the contract
underneath the code. A renamed keyword raises loudly; a changed *default* or a
changed *semantic* (max_seqlen_kv=-1, mask_mode=3, win_left/right=-1) does not.

This script does not touch the NPU: it imports and introspects only.

Usage:
    python3 scripts/checks/qfa_op_contract.py                       # report
    python3 scripts/checks/qfa_op_contract.py --save qfa_base.json  # snapshot
    python3 scripts/checks/qfa_op_contract.py --diff qfa_base.json  # vs snapshot

Take the snapshot on a CANN build that is known to serve correctly, then --diff
after every CANN swap. RED means the contract moved.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

# Keyword arguments attention_v1.py actually passes. Keep in sync with
# _get_qfa_metadata() and _run_qfa(); a keyword that disappears from the
# delivery is a hard break, one that gains a new default is a silent one.
METADATA_KWARGS = [
    "cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv", "v_descale",
    "max_seqlen_q", "max_seqlen_kv", "mask_mode", "win_left", "win_right",
    "layout_q", "layout_q_descale", "layout_kv", "layout_out",
]
MAIN_KWARGS = [
    "block_table", "p_scale", "cu_seqlens_q", "cu_seqlens_kv", "seqused_q",
    "seqused_kv", "sinks", "attn_mask", "metadata", "softmax_scale",
    "mask_mode", "win_left", "win_right", "max_seqlen_q", "max_seqlen_kv",
    "layout_q", "layout_q_descale", "layout_kv", "layout_out",
    "return_softmax_lse",
]
# Positional arguments the code relies on, in order.
METADATA_POSITIONAL = 4  # num_heads, num_kv_heads, head_size, quant_mode
MAIN_POSITIONAL = 7      # q, k, v, q_descale, k_descale, v_descale, quant_mode

# torch_npu entry points the C8_MXFP path depends on.
TORCH_NPU_OPS = ["npu_dynamic_mx_quant", "npu_quantize", "npu_scatter_nd_update_"]


def cann_version() -> dict[str, str]:
    out: dict[str, str] = {}
    root = os.environ.get("ASCEND_TOOLKIT_HOME") or "/usr/local/Ascend/ascend-toolkit/latest"
    for name in ("version.cfg", "ascend_toolkit_install.info"):
        path = Path(root) / name
        if path.is_file():
            try:
                out[name] = path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError as exc:
                out[name] = f"<unreadable: {exc}>"
    out["ASCEND_TOOLKIT_HOME"] = root
    return out


def describe(fn: Any) -> dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        return {"error": f"no introspectable signature: {exc}"}
    params = {}
    for name, param in sig.parameters.items():
        params[name] = {
            "kind": str(param.kind),
            "default": "<required>" if param.default is inspect.Parameter.empty else repr(param.default),
        }
    return {"params": params, "text": str(sig)}


def collect() -> dict[str, Any]:
    report: dict[str, Any] = {"cann": cann_version(), "ops": {}, "torch_npu": {}}

    try:
        import torch  # noqa: F401
        import torch_npu

        report["torch_npu"]["version"] = getattr(torch_npu, "__version__", "<unknown>")
        e8m0 = getattr(torch_npu, "float8_e8m0fnu", None)
        # attention_v1.py notes this is the integer dtype ID (293) on the
        # build it was written against, NOT a torch.dtype. Record which it is.
        report["torch_npu"]["float8_e8m0fnu"] = {"repr": repr(e8m0), "type": type(e8m0).__name__}
        for name in TORCH_NPU_OPS:
            report["torch_npu"][name] = "present" if hasattr(torch_npu, name) else "MISSING"
    except Exception as exc:  # noqa: BLE001
        report["torch_npu"]["error"] = f"{type(exc).__name__}: {exc}"

    try:
        from cann_ops_transformer.ops import quant_flash_attn, quant_flash_attn_metadata

        import cann_ops_transformer

        report["ops"]["package_version"] = getattr(cann_ops_transformer, "__version__", "<unknown>")
        report["ops"]["package_path"] = getattr(cann_ops_transformer, "__file__", "<unknown>")
        report["ops"]["quant_flash_attn"] = describe(quant_flash_attn)
        report["ops"]["quant_flash_attn_metadata"] = describe(quant_flash_attn_metadata)
    except Exception as exc:  # noqa: BLE001
        report["ops"]["error"] = f"{type(exc).__name__}: {exc}"

    return report


def check(report: dict[str, Any]) -> bool:
    ok = True
    print("=== CANN / torch_npu ===")
    for key, value in report["cann"].items():
        print(f"  {key}: {value.splitlines()[0] if value else value}")
    for key, value in report["torch_npu"].items():
        print(f"  torch_npu.{key}: {value}")
    if report["torch_npu"].get("error"):
        print("[RED ] torch_npu unavailable -- run this on the server, inside the container")
        return False
    for name in TORCH_NPU_OPS:
        if report["torch_npu"].get(name) == "MISSING":
            print(f"[RED ] torch_npu.{name} is gone; the C8_MXFP path calls it")
            ok = False

    print("\n=== cann_ops_transformer QFA delivery ===")
    if report["ops"].get("error"):
        print(f"[RED ] {report['ops']['error']}")
        return False
    print(f"  package: {report['ops']['package_version']} @ {report['ops']['package_path']}")

    for op_name, expected_kwargs, n_positional in (
        ("quant_flash_attn_metadata", METADATA_KWARGS, METADATA_POSITIONAL),
        ("quant_flash_attn", MAIN_KWARGS, MAIN_POSITIONAL),
    ):
        info = report["ops"][op_name]
        print(f"\n  {op_name}{info.get('text', '')}")
        if "error" in info:
            print(f"[RED ] {op_name}: {info['error']}")
            ok = False
            continue
        params = info["params"]
        names = list(params)
        missing = [k for k in expected_kwargs if k not in params]
        if missing:
            print(f"[RED ] {op_name}: keywords the code passes are gone: {missing}")
            ok = False
        else:
            print(f"[GREEN] {op_name}: all {len(expected_kwargs)} passed keywords still accepted")
        if len(names) < n_positional:
            print(f"[RED ] {op_name}: fewer than {n_positional} leading positional params")
            ok = False
        else:
            print(f"  leading positional: {names[:n_positional]}")
    return ok


def diff(report: dict[str, Any], baseline_path: Path) -> bool:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    ok = True
    print(f"\n=== diff vs {baseline_path} ===")

    for label, new, old in (
        ("cann", report["cann"], baseline.get("cann", {})),
        ("torch_npu", report["torch_npu"], baseline.get("torch_npu", {})),
    ):
        for key in sorted(set(new) | set(old)):
            if new.get(key) != old.get(key):
                print(f"[info] {label}.{key}: {old.get(key)!r} -> {new.get(key)!r}")

    for op_name in ("quant_flash_attn_metadata", "quant_flash_attn"):
        new_params = report["ops"].get(op_name, {}).get("params", {})
        old_params = baseline.get("ops", {}).get(op_name, {}).get("params", {})
        if not old_params:
            print(f"[info] {op_name}: no baseline recorded")
            continue
        for key in sorted(set(new_params) | set(old_params)):
            if key not in old_params:
                print(f"[info] {op_name}: new parameter {key} = {new_params[key]}")
            elif key not in new_params:
                print(f"[RED ] {op_name}: parameter {key} removed (was {old_params[key]})")
                ok = False
            elif new_params[key] != old_params[key]:
                print(f"[RED ] {op_name}.{key}: {old_params[key]} -> {new_params[key]}")
                ok = False
    if ok:
        print("[GREEN] operator contract unchanged since the baseline")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", metavar="PATH", help="write the collected report as a JSON baseline")
    parser.add_argument("--diff", metavar="PATH", help="compare the collected report against a JSON baseline")
    args = parser.parse_args()

    report = collect()
    ok = check(report)

    if args.save:
        Path(args.save).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\n[info] baseline written to {args.save}")
    if args.diff:
        ok &= diff(report, Path(args.diff))

    print(f"\n{'[GREEN] contract check passed' if ok else '[RED ] contract check failed'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
