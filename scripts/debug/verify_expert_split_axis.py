#!/usr/bin/env python3
"""Prove which axis the mxfp8 export used when splitting fused gate_up experts.

Shape evidence from compare_checkpoint_shapes.py:

  bf16      model.layers.N.mlp.experts.gate_up_proj   [512, 4096, 8192]
            -> one expert is [4096, 8192] = [2 * moe_intermediate, hidden]
  quantized model.layers.N.mlp.experts.N.gate_proj.weight   [4096, 4096]
            model.layers.N.mlp.experts.N.up_proj.weight     [4096, 4096]

[4096, 4096] is the fused [4096, 8192] cut in half along dim 1 (the HIDDEN
axis) instead of dim 0 (the gate/up axis).  If that is what happened, then

    gate_proj == fused[:, :4096]     and     up_proj == fused[:, 4096:]

i.e. each file holds half of hidden for BOTH gate and up, rather than all of
hidden for one of them.

This checks it on the real bytes without dequantizing anything: FP8-E4M3 and
BF16 both keep the sign in the top bit, so the sign pattern must match
element-for-element under the correct hypothesis (~100%) and be uncorrelated
under a wrong one (~50%).  Zeros are excluded since their sign is meaningless.

Needs numpy. Reads a few hundred KB of tensor payload via seek, imports no
torch, never touches the NPU.

Usage:
  python3 verify_expert_split_axis.py [QUANT_PATH] [BF16_PATH]
                                      [--layer L] [--expert E] [--rows R]
Final line is [GREEN] (hypothesis confirmed) or [RED].
"""

import json
import re
import struct
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("this script needs numpy (pip install numpy)")

DEFAULT_QUANT_PATH = "/mnt/share/weight/Qwen3.8-2.4T-A95B-mxfp8"
GATE_RE = re.compile(r"\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.weight$")
FUSED_RE = re.compile(r"\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|w13)$")

DTYPE_BYTES = {"F8_E4M3": 1, "F8_E5M2": 1, "U8": 1, "I8": 1,
               "BF16": 2, "F16": 2, "F32": 4}


def read_header(path):
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(header_len)), 8 + header_len


def read_rows(path, entry, data_start, expert, rows):
    """Read `rows` leading rows of one expert slice as a flat uint8 array."""
    shape = entry["shape"]
    itemsize = DTYPE_BYTES.get(entry["dtype"])
    if itemsize is None:
        sys.exit("unsupported dtype " + entry["dtype"])
    if len(shape) == 3:
        expert_elems = shape[1] * shape[2]
        row_elems = shape[2]
        skip = expert * expert_elems
    else:
        row_elems = shape[1]
        skip = 0
    count = min(rows, shape[-2]) * row_elems
    offset = data_start + entry["data_offsets"][0] + skip * itemsize
    with open(path, "rb") as fh:
        fh.seek(offset)
        buf = fh.read(count * itemsize)
    if len(buf) < count * itemsize:
        sys.exit("short read on " + str(path))
    return np.frombuffer(buf, dtype=np.uint8), row_elems, itemsize


def signs(raw, row_elems, itemsize, rows):
    """Top-bit sign plane plus a mask of non-zero elements, shaped [rows, cols].

    Little-endian for every float format here, so the sign is the top bit of the
    last byte and the remaining bits (plus the lower bytes) give the magnitude.
    """
    b = raw.reshape(-1, itemsize)
    hi = b[:, itemsize - 1]
    sign = (hi >> 7).reshape(rows, row_elems)
    nonzero = (((hi & 0x7F) != 0) | b[:, : itemsize - 1].any(axis=1)).reshape(rows, row_elems)
    return sign, nonzero


def agreement(sign_a, ok_a, sign_b, ok_b):
    mask = ok_a & ok_b
    if not mask.any():
        return float("nan"), 0
    return float((sign_a[mask] == sign_b[mask]).mean()), int(mask.sum())


def find_quant_target(quant_dir, want_layer, want_expert):
    """Scan shards until a per-expert gate/up pair from one expert is found."""
    for shard in sorted(quant_dir.glob("*.safetensors")):
        header, data_start = read_header(shard)
        for name in header:
            m = GATE_RE.search(name)
            if not m:
                continue
            layer, expert = int(m.group(1)), int(m.group(2))
            if want_layer is not None and layer != want_layer:
                continue
            if want_expert is not None and expert != want_expert:
                continue
            up = name.replace(".gate_proj.", ".up_proj.")
            if up in header:
                return shard, header, data_start, name, up, layer, expert
    return None


def main():
    argv, positional = list(sys.argv[1:]), []
    layer = expert = None
    rows = 64
    while argv:
        arg = argv.pop(0)
        if arg == "--layer":
            layer = int(argv.pop(0))
        elif arg == "--expert":
            expert = int(argv.pop(0))
        elif arg == "--rows":
            rows = int(argv.pop(0))
        elif arg.startswith("-"):
            sys.exit("unknown flag " + arg)
        else:
            positional.append(arg)

    quant_dir = Path(positional[0] if positional else DEFAULT_QUANT_PATH)
    if len(positional) > 1:
        bf16_dir = Path(positional[1])
    else:
        name = quant_dir.name
        bf16_dir = quant_dir.parent / (name[: -len("-mxfp8")] if name.endswith("-mxfp8") else name)
    print("quantized = " + str(quant_dir))
    print("bf16      = " + str(bf16_dir))

    found = find_quant_target(quant_dir, layer, expert)
    if found is None:
        print("[RED] no per-expert gate/up pair found in the quantized checkpoint")
        return 1
    q_shard, q_header, q_start, gate_name, up_name, layer, expert = found
    print("")
    print("probe: layer " + str(layer) + ", expert " + str(expert))
    print("  " + gate_name + "  " + str(q_header[gate_name]["shape"])
          + " " + q_header[gate_name]["dtype"] + "  (" + q_shard.name + ")")
    print("  " + up_name + "  " + str(q_header[up_name]["shape"])
          + " " + q_header[up_name]["dtype"])

    index_path = bf16_dir / "model.safetensors.index.json"
    fused_name = fused_shard = None
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        for name, filename in weight_map.items():
            m = FUSED_RE.search(name)
            if m and int(m.group(1)) == layer:
                fused_name, fused_shard = name, bf16_dir / filename
                break
    if fused_name is None:
        for shard in sorted(bf16_dir.glob("*.safetensors")):
            header, _ = read_header(shard)
            for name in header:
                m = FUSED_RE.search(name)
                if m and int(m.group(1)) == layer:
                    fused_name, fused_shard = name, shard
                    break
            if fused_name:
                break
    if fused_name is None:
        print("[RED] bf16 checkpoint has no fused experts tensor for layer " + str(layer))
        return 1
    b_header, b_start = read_header(fused_shard)
    fused_entry = b_header[fused_name]
    print("  " + fused_name + "  " + str(fused_entry["shape"])
          + " " + fused_entry["dtype"] + "  (" + fused_shard.name + ")")

    gate_entry, up_entry = q_header[gate_name], q_header[up_name]
    q_rows = min(rows, gate_entry["shape"][0], fused_entry["shape"][1])
    half = fused_entry["shape"][2] // 2
    if gate_entry["shape"][1] != half:
        print("[RED] quantized gate width " + str(gate_entry["shape"][1])
              + " is not half of hidden " + str(fused_entry["shape"][2])
              + "; this script only tests the wrong-axis hypothesis")
        return 1

    raw_g, cols_g, item_g = read_rows(q_shard, gate_entry, q_start, 0, q_rows)
    raw_u, _, item_u = read_rows(q_shard, up_entry, q_start, 0, q_rows)
    raw_f, cols_f, item_f = read_rows(fused_shard, fused_entry, b_start, expert, q_rows)

    sign_g, ok_g = signs(raw_g, cols_g, item_g, q_rows)
    sign_u, ok_u = signs(raw_u, cols_g, item_u, q_rows)
    sign_f, ok_f = signs(raw_f, cols_f, item_f, q_rows)

    left = (sign_f[:, :half], ok_f[:, :half])
    right = (sign_f[:, half:], ok_f[:, half:])

    print("")
    print("sign agreement over " + str(q_rows) + " rows (100% = same data, ~50% = unrelated):")
    print("  " + "".ljust(12) + "fused[:, :" + str(half) + "]".ljust(18) + "fused[:, " + str(half) + ":]")
    results = {}
    for label, (sign_q, ok_q) in (("gate_proj", (sign_g, ok_g)), ("up_proj", (sign_u, ok_u))):
        row = []
        for side, (sign_f_half, ok_f_half) in (("left", left), ("right", right)):
            rate, n = agreement(sign_q, ok_q, sign_f_half, ok_f_half)
            results[(label, side)] = rate
            row.append("{:.1%}".format(rate) + " (n=" + str(n) + ")")
        print("  " + label.ljust(12) + row[0].ljust(18) + row[1])

    print("")
    strong = 0.95
    weak = 0.65
    gate_left, gate_right = results[("gate_proj", "left")], results[("gate_proj", "right")]
    up_left, up_right = results[("up_proj", "left")], results[("up_proj", "right")]
    if gate_left > strong and up_right > strong and gate_right < weak and up_left < weak:
        print("[GREEN] confirmed: gate_proj IS fused[:, :" + str(half) + "] and up_proj IS fused[:, "
              + str(half) + ":].")
        print("        The export split the fused gate_up tensor along dim 1 (hidden) instead of")
        print("        dim 0 (gate/up). Each 'gate_proj' file holds the first half of hidden for")
        print("        BOTH gate and up, so the names are wrong and the tensors are unusable as-is.")
        print("        Fix belongs in the ModelSlim export, not in vLLM or vllm-ascend.")
        return 0
    if max(gate_left, gate_right, up_left, up_right) < weak:
        print("[RED] no correspondence found - the quantized tensors are not a plain slice of the")
        print("      bf16 fused tensor (a permutation or extra transform is involved).")
        return 1
    print("[RED] partial/unexpected correspondence; read the table above before concluding.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
