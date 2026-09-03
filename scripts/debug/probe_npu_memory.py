#!/usr/bin/env python3
"""Print the HBM totals torch actually sees, without loading a model.

Every KV-cache budget starts from

  requested_memory = init_snapshot.total_memory * gpu_memory_utilization

(worker.py:443), and `total_memory` comes from torch, not from npu-smi.  The
two disagree: npu-smi reports the physical HBM (98304 MiB on an A5), while
torch reports what is left after the driver's own reservation.  Planning
against the npu-smi number silently overstates the budget.

This is the only script in debug/ that touches the NPU, and it is deliberately
minimal: it creates an ACL context (a few hundred MiB, released on exit), reads
mem_get_info, and returns.  No weights, no tensors, no collectives.  Safe to
run on a machine you do not have reserved, but it does briefly occupy the
device, so leave --all off unless you need the per-device spread.

Usage:
  python3 probe_npu_memory.py [--device 0] [--all] [--util 0.85 --util 0.93]
"""

import argparse
import sys

GIB = float(1 << 30)


def probe(device):
    """Return (free, total) HBM bytes for one device; creates its ACL context."""
    import torch

    return torch.npu.mem_get_info(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="probe every visible device")
    parser.add_argument("--util", type=float, action="append", default=None,
                        help="utilization values to tabulate; repeatable")
    args = parser.parse_args()

    try:
        import torch
        import torch_npu  # noqa: F401  (registers the npu backend on import)
    except ImportError as exc:
        print("[RED] cannot import torch/torch_npu: " + str(exc))
        return 1

    if not torch.npu.is_available():
        print("[RED] torch.npu.is_available() is False")
        return 1

    utils = args.util or [0.85, 0.90, 0.93, 0.95]
    devices = range(torch.npu.device_count()) if args.all else [args.device]

    print("=== HBM as torch sees it ===")
    print("  %-8s %10s %10s %10s" % ("device", "total", "free", "used"))
    readings = []
    for device in devices:
        free, total = probe(device)
        readings.append((device, free, total))
        print("  npu:%-4d %8.2f G %8.2f G %8.2f G" % (device, total / GIB, free / GIB, (total - free) / GIB))

    reference = min(total for _, _, total in readings)
    print("=== requested_memory = total x util ===")
    for util in utils:
        print("  util=%.2f -> %7.2f GiB" % (util, reference * util / GIB))
    print("  (feed this total to plan_hbm_budget.py as --hbm-gib %.2f)" % (reference / GIB))

    # Our own ACL context is part of `used`, so only a large residue means
    # someone else's process is still holding the device.
    busy = [str(d) for d, free, total in readings if free < total * 0.9]
    if busy:
        print("[RED] devices %s still hold >10%% of HBM; clear them with npu-cleaner.sh before planning"
              % ", ".join(busy))
        return 1
    print("[GREEN] devices idle; the totals above are the planning baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
