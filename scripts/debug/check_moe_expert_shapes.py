#!/usr/bin/env python3
"""Explain the MoE w13 load failure by comparing checkpoint layout to config.json.

Context: serving the 2.4T checkpoint dies during weight load with

  File ".../fused_moe/routed_experts.py", line 529, in _load_w13
    expert_data.copy_(loaded_weight)
  RuntimeError: The expanded size of the tensor (2048) must match the existing
  size (4096) at non-singleton dimension 0.
  Target sizes: [2048, 4096].  Tensor sizes: [4096, 4096].

vLLM's RoutedExperts.load_weights supports BOTH expert layouts, but picks the
branch purely from the checkpoint tensor NAME (routed_experts.py:892):

    is_per_expert_fused_w13 = (
        not is_fused                                   # 2D tensor
        and shard_id in {"w1", "w3"}
        and any(f".{n}." in qual_name for n in ("gate_up_proj", "w13"))
    )

  * name contains ".gate_up_proj." / ".w13."  -> chunk(2, dim=0), gate & up split
  * otherwise                                 -> the whole tensor is copied as w1

So a checkpoint that stores FUSED gate+up rows under a NON-fused name
(".gate_proj.") -- or a config.json whose moe_intermediate_size is half the real
value -- lands in the second branch and fails with exactly a 2x mismatch.

This script reads only safetensors headers (8-byte length prefix + JSON, no
tensor payload), never imports torch/vllm, and never touches the NPU.

Usage: python3 check_moe_expert_shapes.py [MODEL_PATH] [--tp 8] [--no-ep]
Final line is [GREEN] (exit 0) or [RED] (exit 1).
"""

import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path

DEFAULT_MODEL_PATH = "/mnt/share/weight/Qwen3.8-2.4T-A95B-mxfp8"

EXPERT_RE = re.compile(
    r"\.layers\.(?P<layer>\d+)\..*experts\.(?P<expert>\d+)\.(?P<proj>[^.]+)\.(?P<suffix>.+)$"
)
# Names vLLM treats as a pre-fused per-expert gate+up tensor.
FUSED_NAMES = ("gate_up_proj", "w13")
GATE_UP_NAMES = FUSED_NAMES + ("gate_proj", "up_proj", "w1", "w3")


def read_safetensors_header(path):
    """Return the safetensors JSON header without reading any tensor payload."""
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(header_len))


def parse_args(argv):
    model_path, tp_size, enable_ep = DEFAULT_MODEL_PATH, 8, True
    rest = list(argv)
    while rest:
        arg = rest.pop(0)
        if arg == "--tp":
            tp_size = int(rest.pop(0))
        elif arg == "--no-ep":
            enable_ep = False
        elif arg.startswith("-"):
            sys.exit("unknown flag " + arg + "; usage: [MODEL_PATH] [--tp N] [--no-ep]")
        else:
            model_path = arg
    return Path(model_path), tp_size, enable_ep


def locate_expert_tensors(model_dir):
    """Map expert tensor name -> shard file, preferring the index file."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        print("index: " + index_path.name + " (" + str(len(weight_map)) + " tensors)")
        return {n: model_dir / f for n, f in weight_map.items() if EXPERT_RE.search(n)}

    shards = sorted(model_dir.glob("*.safetensors"))
    print("index: absent; scanning " + str(len(shards)) + " shard header(s) directly")
    found = {}
    for shard in shards:
        for name in read_safetensors_header(shard):
            if EXPERT_RE.search(name):
                found[name] = shard
        if found:
            # One shard's worth of experts is enough to judge the layout.
            break
    return found


def main():
    model_dir, tp_size, enable_ep = parse_args(sys.argv[1:])
    print("model_path      = " + str(model_dir))
    print("assumed runtime = TP" + str(tp_size) + ", expert-parallel " + ("ON" if enable_ep else "OFF"))

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        print("[RED] missing " + str(config_path))
        return 1
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", config)

    hidden_size = text_config.get("hidden_size")
    moe_intermediate = text_config.get("moe_intermediate_size")
    print("model_type      = " + str(config.get("model_type")))
    print("hidden_size     = " + str(hidden_size))
    print("moe_intermediate_size = " + str(moe_intermediate))
    print("num_experts     = " + str(text_config.get("num_experts", text_config.get("num_local_experts"))))
    if not hidden_size or not moe_intermediate:
        print("[RED] config.json lacks hidden_size / moe_intermediate_size; cannot judge layout")
        return 1

    expert_files = locate_expert_tensors(model_dir)
    if not expert_files:
        print("[RED] no '...experts.<id>.<proj>.<suffix>' tensors found; layout is unexpected")
        return 1

    proj_names = Counter(EXPERT_RE.search(n).group("proj") for n in expert_files)
    print("")
    print("expert projection names seen (" + str(len(expert_files)) + " tensors):")
    for proj, count in sorted(proj_names.items(), key=lambda kv: -kv[1]):
        note = "   <- vLLM splits this one as fused gate+up" if proj in FUSED_NAMES else ""
        print("  " + proj.ljust(16) + " x" + str(count) + note)

    weights = [n for n in expert_files if n.endswith(".weight")]
    if not weights:
        print("[RED] expert tensors found but none end in '.weight'")
        return 1

    def sort_key(name):
        m = EXPERT_RE.search(name)
        return (int(m.group("layer")), int(m.group("expert")), m.group("proj"))

    sample = EXPERT_RE.search(min(weights, key=sort_key))
    layer, expert = sample.group("layer"), sample.group("expert")
    peers = {}
    for name in weights:
        m = EXPERT_RE.search(name)
        if m.group("layer") == layer and m.group("expert") == expert:
            peers[name] = expert_files[name]

    print("")
    print("layer " + layer + " / expert " + expert + " tensors (from safetensors headers):")
    headers, shapes = {}, {}
    for name in sorted(peers):
        shard = peers[name]
        if shard not in headers:
            headers[shard] = read_safetensors_header(shard)
        entry = headers[shard].get(name)
        if entry is None:
            print("  " + name + "  [RED] listed in index but absent from " + shard.name)
            return 1
        shapes[EXPERT_RE.search(name).group("proj")] = entry["shape"]
        print("  " + name)
        print("      shape=" + str(entry["shape"]) + " dtype=" + entry["dtype"] + " file=" + shard.name)

    # Reproduce how vLLM sizes w13, then how it would copy each checkpoint tensor.
    moe_tp_size = 1 if enable_ep else tp_size
    inter_per_partition = moe_intermediate // moe_tp_size
    print("")
    print("vLLM builds the MoE layer as:")
    print("  moe tp_size                     = " + str(moe_tp_size)
          + (" (EP on: experts split by rank, rows kept whole)" if enable_ep
             else " (EP off: rows split across TP)"))
    print("  intermediate_size_per_partition = " + str(moe_intermediate) + " // " + str(moe_tp_size)
          + " = " + str(inter_per_partition))
    print("  w13_weight[expert].shape        = [" + str(2 * inter_per_partition) + ", " + str(hidden_size) + "]")
    print("  _load_w13 target slice          = [" + str(inter_per_partition) + ", " + str(hidden_size) + "]")

    gate_like = sorted(p for p in shapes if p in GATE_UP_NAMES)
    if not gate_like:
        print("")
        print("[RED] no gate/up style expert tensor found; cannot reproduce the failing path")
        return 1

    failures = []
    print("")
    print("simulating RoutedExperts.load_weights:")
    for proj in gate_like:
        rows, cols = shapes[proj][0], shapes[proj][-1]
        fused = proj in FUSED_NAMES
        effective = rows // 2 if fused else rows
        if moe_tp_size > 1:
            effective //= moe_tp_size
        branch = "chunk(2, dim=0), gate/up split" if fused else "whole tensor copied as one shard"
        ok = effective == inter_per_partition and cols == hidden_size
        print("  " + proj.ljust(16) + " shape=" + str(shapes[proj]) + "  branch: " + branch)
        print("  " + "".ljust(16) + " -> copy_ target [" + str(inter_per_partition) + ", " + str(hidden_size)
              + "]  <-  tensor [" + str(effective) + ", " + str(cols) + "]   "
              + ("ok" if ok else "MISMATCH"))
        if not ok:
            failures.append((proj, shapes[proj], effective, fused))

    down = shapes.get("down_proj") or shapes.get("w2")
    if down:
        ok = list(down) == [hidden_size, inter_per_partition]
        print("  " + "down_proj".ljust(16) + " shape=" + str(down) + "  expected ["
              + str(hidden_size) + ", " + str(inter_per_partition) + "]   " + ("ok" if ok else "MISMATCH"))
        if not ok:
            failures.append(("down_proj", down, down[0], False))

    print("")
    if not failures:
        print("[GREEN] checkpoint expert layout matches config.json and the runtime parallel plan.")
        print("        The w13 mismatch comes from elsewhere - suspect the quantization method")
        print("        or a vllm-ascend patch resizing w13, not the checkpoint itself.")
        return 0

    proj, shape, effective, fused = failures[0]
    print("[RED] expert tensor '" + proj + "' does not fit the layer vLLM built from config.json.")
    if not fused and shape[0] == 2 * moe_intermediate:
        print("      It has " + str(shape[0]) + " rows = 2 x moe_intermediate_size, so gate and up are")
        print("      FUSED in the checkpoint, but the name '" + proj + "' is not one of "
              + str(list(FUSED_NAMES)) + ",")
        print("      so vLLM copies the whole tensor as a single shard -> the 2x mismatch you saw.")
        print("      This is a CHECKPOINT/NAMING problem, not a vLLM or vllm-ascend bug.")
    elif shape[0] != inter_per_partition * (2 if fused else 1):
        print("      Row count " + str(shape[0]) + " disagrees with moe_intermediate_size="
              + str(moe_intermediate) + ".")
        print("      config.json and these safetensors were most likely produced by different runs.")
    else:
        print("      Column count " + str(shape[-1]) + " disagrees with hidden_size=" + str(hidden_size) + ".")
    return 1


if __name__ == "__main__":
    sys.exit(main())
