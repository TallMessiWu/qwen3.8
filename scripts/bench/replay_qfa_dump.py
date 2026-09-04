#!/usr/bin/env python3
"""Feed a QFA dump back into the operator and check it reproduces the same output.

Different question from analyze_qfa_dump.py, which never calls QFA at all: that
one recomputes attention in float64 to measure error, this one re-runs the real
kernel on the real inputs. What it establishes:

  loadable    the dump carries every argument the operator needs, so it can be
              replayed away from a live server -- which is what an operator bug
              report has to ship.
  runnable    those arguments are self-consistent: shapes, layouts, the sliced
              cache and its renumbered block table all still satisfy the kernel.
  identical   the replayed output matches what was recorded. Same inputs, same
              kernel, same device should be bit-exact; anything else means the
              result depends on state the dump did not capture, which is a
              finding in itself rather than a rounding detail.

It also re-derives the AICPU metadata from op_kwargs and replays with that, so a
mismatch tells you whether the plan is reproducible or carries run-specific
state -- worth knowing before quoting a plan header in a bug report.

Needs an NPU. Use --dry-run to check loading, dtype restoration and shape
reconstruction anywhere, without torch_npu and without touching the device.

Usage:
  python3 replay_qfa_dump.py DUMP_DIR [--dry-run]
Final line is [GREEN] (exit 0) or [RED] (exit 1).
"""

import argparse
import sys
from pathlib import Path

import torch


def load_dump(path: str) -> dict:
    """Read a dump back, restoring FP8 dtypes that travelled as uint8 byte views."""

    def restore(obj):
        if isinstance(obj, dict):
            if "__fp8_dtype__" in obj:
                name = obj["__fp8_dtype__"].removeprefix("torch.")
                dtype = getattr(torch, name, None)
                return obj["bytes"] if dtype is None else obj["bytes"].view(dtype)
            return {k: restore(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(restore(v) for v in obj)
        return obj

    return restore(torch.load(path, map_location="cpu", weights_only=False))


def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        # FP8 moves as a byte view: transpose/copy on float8 either errors or
        # falls back to AICPU on NPU.
        if "float8" in str(obj.dtype):
            return obj.view(torch.uint8).to(device).view(obj.dtype)
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj


def describe(name: str, tensor) -> str:
    if not isinstance(tensor, torch.Tensor):
        return f"    {name:<22} {tensor!r}"
    return f"    {name:<22} {tuple(tensor.shape)!s:<28} {tensor.dtype}"


def check_shapes(call: dict) -> list[str]:
    """Structural checks that must hold for the replay to even be meaningful."""
    problems = []
    key_cache = call["key_cache"]
    value_cache = call["value_cache"]
    key_scale_cache = call["key_scale_cache"]
    value_scale_cache = call["value_scale_cache"]
    block_table = call["block_table_local"]
    nb, bs, n, d = key_cache.shape

    if value_cache.shape != key_cache.shape:
        problems.append(f"value_cache {tuple(value_cache.shape)} != key_cache {tuple(key_cache.shape)}")
    expected_k_scale = (nb, bs, n, d // 64, 2)
    if tuple(key_scale_cache.shape) != expected_k_scale:
        problems.append(f"key_scale_cache {tuple(key_scale_cache.shape)} != expected {expected_k_scale}")
    expected_v_scale = (nb, bs // 64, n, d, 2)
    if tuple(value_scale_cache.shape) != expected_v_scale:
        problems.append(f"value_scale_cache {tuple(value_scale_cache.shape)} != expected {expected_v_scale}")

    # The renumbered table must address only blocks present in the slice.
    max_block = int(block_table.max()) if block_table.numel() else -1
    if max_block >= nb:
        problems.append(f"block_table_local addresses block {max_block} but the slice has {nb}")

    # Every sequence must fit inside the blocks its row provides.
    seqused_kv = call["op_kwargs"]["seqused_kv"]
    for row in range(min(block_table.shape[0], seqused_kv.numel())):
        kv_len = int(seqused_kv[row])
        needed = (kv_len + bs - 1) // bs
        if needed > block_table.shape[1]:
            problems.append(f"sequence {row} needs {needed} blocks, table row has {block_table.shape[1]}")
    return problems


def replay_one(write_file: Path, call_file: Path, dump_dir: Path, dry_run: bool) -> bool:
    label = call_file.name.replace("__qfa_call.pt", "")
    print(f"\n=== {label} ===")
    call = load_dump(str(call_file))

    print("  recorded arguments:")
    for name in ("q_fp8", "q_descale", "key_cache", "value_cache", "key_scale_cache", "value_scale_cache"):
        print(describe(name, call[name]))
    print(describe("block_table_local", call["block_table_local"]))
    print(describe("metadata", call["metadata"]))
    print(describe("attn_output", call["attn_output"]))
    op_kwargs = call["op_kwargs"]
    scalars = {k: v for k, v in op_kwargs.items() if not isinstance(v, torch.Tensor)}
    print(f"    op_kwargs scalars      {scalars}")

    problems = check_shapes(call)
    if problems:
        for problem in problems:
            print(f"  [FAIL] {problem}")
        return False
    print("  shape/layout checks passed")

    mask_file = call.get("attn_mask_file")
    attn_mask = None
    if mask_file:
        mask_path = dump_dir / mask_file
        if not mask_path.is_file():
            print(f"  [FAIL] attn_mask file {mask_file} missing from the dump directory")
            return False
        attn_mask = load_dump(str(mask_path))
        print(describe("attn_mask", attn_mask))

    if dry_run:
        print("  dry run: everything needed for a replay is present and consistent")
        return True

    device = "npu"
    q_fp8 = to_device(call["q_fp8"], device)
    q_descale = to_device(call["q_descale"], device)
    key = to_device(call["key_cache"], device)
    value = to_device(call["value_cache"], device)
    key_scale = to_device(call["key_scale_cache"], device)
    value_scale = to_device(call["value_scale_cache"], device)
    block_table = to_device(call["block_table_local"], device)
    metadata = to_device(call["metadata"], device)
    op_kwargs_dev = to_device(op_kwargs, device)
    mask_dev = to_device(attn_mask, device) if attn_mask is not None else None
    recorded = call["attn_output"]

    def run(meta):
        out, _ = torch.ops._C_ascend.npu_quant_flash_attn(
            q_fp8,
            key,
            value,
            q_descale,
            key_scale.view(torch.float8_e8m0fnu),
            value_scale.view(torch.float8_e8m0fnu),
            1,
            block_table=block_table,
            attn_mask=mask_dev,
            metadata=meta,
            softmax_scale=float(call["softmax_scale"]),
            **op_kwargs_dev,
        )
        return out

    ok = True

    # --- replay with the recorded plan ----------------------------------
    try:
        out = run(metadata).cpu().reshape(recorded.shape)
    except Exception as exc:  # noqa: BLE001 - the failure itself is the result
        print(f"  [FAIL] replay with the recorded metadata raised: {type(exc).__name__}: {exc}")
        return False
    same = torch.equal(out.to(torch.float64), recorded.to(torch.float64))
    diff = float(
        torch.linalg.vector_norm(out.to(torch.float64) - recorded.to(torch.float64))
        / torch.linalg.vector_norm(recorded.to(torch.float64)).clamp(min=1e-30)
    )
    print(f"  replay (recorded plan):    bit-exact={same}  rel_l2={diff:.3e}")
    if not same:
        # Not a rounding question: identical inputs through the same kernel
        # should be deterministic, so any difference means hidden state.
        print("  [FAIL] output differs from what was recorded with identical inputs")
        ok = False

    # --- replay with a freshly derived plan ------------------------------
    try:
        fresh = torch.ops._C_ascend.npu_quant_flash_attn_metadata(
            int(call["num_heads"]),
            int(call["num_kv_heads"]),
            int(call["head_size"]),
            1,
            # Same 5D minimum-size E8M0 placeholder the builder uses: the aclnn
            # entry requires vDescale to be non-null and 5D under PA_BBND, but
            # nothing reads it there.
            v_descale=torch.zeros(1, 1, 1, 1, 2, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu),
            **op_kwargs_dev,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  metadata rebuild raised {type(exc).__name__}: {exc}")
        print("  (skipping the fresh-plan replay; the recorded-plan result above still stands)")
        return ok

    plan_same = torch.equal(fresh.cpu()[:4], metadata.cpu()[:4])
    print(f"  metadata header reproducible: {plan_same}  recorded={metadata.cpu()[:4].tolist()}")
    try:
        out2 = run(fresh).cpu().reshape(recorded.shape)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] replay with a rebuilt plan raised: {type(exc).__name__}: {exc}")
        return False
    same2 = torch.equal(out2.to(torch.float64), recorded.to(torch.float64))
    print(f"  replay (rebuilt plan):     bit-exact={same2}")
    if not same2:
        print("  [WARN] a rebuilt plan gives a different result than the recorded one")

    return ok


def register_custom_ops() -> bool:
    """Register torch.ops._C_ascend.* -- vllm-ascend's kernels, not torch_npu's.

    Deliberately not via enable_custom_op(): that gates on
    get_current_hardware_profile(), which a standalone script has not set up, and
    it swallows the ImportError so a failure arrives as a bare False. Doing the
    two real steps here keeps the actual reason visible.
    """
    import torch_npu  # noqa: F401  (registers the NPU device)
    from vllm_ascend.utils import bootstrap_custom_op_env

    def _import_extension():
        import vllm_ascend.vllm_ascend_C  # type: ignore # noqa: F401

    bootstrap_custom_op_env()
    try:
        _import_extension()
    except ImportError as exc:
        # The extension prefers its own rpath for the vendor op_api; fall back to
        # LD_LIBRARY_PATH only when the import says that is what is missing.
        if "libcust_opapi.so" not in str(exc):
            print(f"  cannot import vllm_ascend.vllm_ascend_C: {exc}")
            return False
        print(f"  first import failed ({exc}); retrying with the vendor lib path")
        bootstrap_custom_op_env(include_vendor_lib=True)
        try:
            _import_extension()
        except ImportError as exc2:
            print(f"  still failing: {exc2}")
            return False

    if not hasattr(torch.ops._C_ascend, "npu_quant_flash_attn"):
        print("  extension loaded but npu_quant_flash_attn is not registered.")
        print("  The QFA kernels are built separately -- see scripts/setup/build_qfa_ops.sh.")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump_dir", help="directory written by VLLM_ASCEND_QFA_DUMP_DIR")
    parser.add_argument("--dry-run", action="store_true", help="load and shape-check only; no NPU needed")
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    if not dump_dir.is_dir():
        print(f"[RED] not a directory: {dump_dir}")
        return 1

    pairs = []
    for call_file in sorted(dump_dir.glob("*__qfa_call.pt")):
        write_file = Path(str(call_file).replace("__qfa_call.pt", "__cache_write.pt"))
        pairs.append((write_file, call_file))
    if not pairs:
        print(f"[RED] no *__qfa_call.pt under {dump_dir}")
        return 1

    mode = "dry run (no NPU)" if args.dry_run else "replaying on NPU"
    print(f"found {len(pairs)} recorded call(s) under {dump_dir} -- {mode}")

    if not args.dry_run:
        print("\nregistering vllm-ascend custom ops...")
        if not register_custom_ops():
            print("\n[RED] cannot replay without torch.ops._C_ascend.npu_quant_flash_attn")
            return 1
        print("  ok, npu_quant_flash_attn is available")

    results = [replay_one(w, c, dump_dir, args.dry_run) for w, c in pairs]

    print(f"\n=== {sum(results)}/{len(results)} call(s) replayed successfully ===")
    if all(results):
        print("[GREEN]")
        return 0
    print("[RED]")
    return 1


if __name__ == "__main__":
    sys.exit(main())
