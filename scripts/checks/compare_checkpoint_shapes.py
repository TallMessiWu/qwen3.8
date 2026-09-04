#!/usr/bin/env python3
"""Diff tensor names/shapes between a quantized checkpoint and its BF16 original.

Context: the mxfp8 2.4T checkpoint fails to load with

  RuntimeError: ... Target sizes: [2048, 8192].  Tensor sizes: [4096, 4096].

because its expert tensors are stored as [4096, 4096] while config.json
(hidden_size=8192, moe_intermediate_size=2048) makes vLLM build [2048, 8192].
Note 4096*4096 == 2048*8192: the element count is preserved, so this is a
LAYOUT change, not a fused gate+up tensor and not a truncated weight.

Comparing against the un-quantized source answers the remaining question:
did the quantization step reshape the weights, or was the layout already
like this before quantizing (i.e. config.json never described these files)?

Reads only safetensors headers (8-byte length prefix + JSON, no tensor
payload), never imports torch/vllm, never touches the NPU.

Usage:
  python3 compare_checkpoint_shapes.py [QUANT_PATH] [BF16_PATH] [--max-shards N]

QUANT_PATH defaults to the mxfp8 directory; BF16_PATH defaults to the same
path with the "-mxfp8" suffix stripped.
Final line is [GREEN] (exit 0) or [RED] (exit 1).
"""

import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_QUANT_PATH = "/mnt/share/weight/Qwen3.8-2.4T-A95B-mxfp8"

# Config keys that decide how vLLM sizes every weight.
CONFIG_KEYS = (
    "model_type", "hidden_size", "intermediate_size", "moe_intermediate_size",
    "num_experts", "num_local_experts", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
)

DIGITS = re.compile(r"\d+")
EXPERT_RE = re.compile(r"\.layers\.(\d+)\..*experts\.(\d+)\.([^.]+)\.(.+)$")


def read_safetensors_header(path):
    """Return the safetensors JSON header without reading any tensor payload."""
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(header_len))


def pattern(name):
    """Collapse every index in a tensor name so shapes can be grouped."""
    return DIGITS.sub("N", name)


def collect(model_dir, label, max_shards):
    """Return {tensor_name: (shape, dtype)} using the index file when present."""
    index_path = model_dir / "model.safetensors.index.json"
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        print("  " + label + ": no *.safetensors under " + str(model_dir))
        return None

    limit = len(shards) if max_shards <= 0 else min(max_shards, len(shards))
    print("  " + label + ": " + str(len(shards)) + " shard(s)"
          + (", index.json present" if index_path.is_file() else ", no index.json")
          + "; reading " + str(limit) + " header(s)")

    tensors = {}
    for i, shard in enumerate(shards[:limit], start=1):
        for name, entry in read_safetensors_header(shard).items():
            if name == "__metadata__":
                continue
            tensors[name] = (tuple(entry["shape"]), entry["dtype"])
        if i % 100 == 0 or i == limit:
            print("    ... " + str(i) + "/" + str(limit) + " shards, "
                  + str(len(tensors)) + " tensors")
    return tensors


def pick_fused_gate_up(tensors):
    """Find a 3-D `experts.gate_up_proj`/`experts.w13` tensor: [E, 2I, H]."""
    for name in sorted(tensors):
        shape = tensors[name][0]
        if len(shape) == 3 and re.search(r"\.experts\.(gate_up_proj|w13)$", name):
            return name, shape
    return None


def pick_per_expert_gate_up(tensors):
    """Find a 2-D per-expert gate tensor: `experts.N.gate_proj.weight`."""
    for name in sorted(tensors):
        shape = tensors[name][0]
        if len(shape) == 2 and re.search(r"\.experts\.\d+\.(gate_proj|w1)\.weight$", name):
            return name, shape
    return None


def load_config(model_dir):
    path = model_dir / "config.json"
    if not path.is_file():
        return None
    config = json.loads(path.read_text())
    return config.get("text_config", config)


def print_config_diff(cfg_a, cfg_b):
    print("")
    print("=== config.json ===")
    print("  " + "key".ljust(24) + "quantized".ljust(20) + "bf16")
    for key in CONFIG_KEYS:
        va, vb = cfg_a.get(key), cfg_b.get(key)
        if va is None and vb is None:
            continue
        flag = "" if va == vb else "   <-- differs"
        print("  " + key.ljust(24) + str(va).ljust(20) + str(vb) + flag)
    for label, cfg in (("quantized", cfg_a), ("bf16", cfg_b)):
        quant = cfg.get("quantization_config")
        if quant:
            keys = sorted(quant) if isinstance(quant, dict) else quant
            print("  " + label + " quantization_config keys: " + str(keys))


def print_pattern_summary(title, names, tensors, limit=12):
    if not names:
        return
    grouped = defaultdict(Counter)
    for name in names:
        grouped[pattern(name)][tensors[name][0]] += 1
    print("")
    print("=== " + title + " (" + str(len(names)) + " tensors, "
          + str(len(grouped)) + " name patterns) ===")
    for pat, shape_counts in sorted(grouped.items(), key=lambda kv: -sum(kv[1].values()))[:limit]:
        total = sum(shape_counts.values())
        shapes = ", ".join(str(list(s)) + " x" + str(c) for s, c in shape_counts.most_common(3))
        print("  " + pat)
        print("      x" + str(total) + "   " + shapes)
    if len(grouped) > limit:
        print("  ... " + str(len(grouped) - limit) + " more pattern(s) not shown")


def main():
    argv = list(sys.argv[1:])
    max_shards, positional = 0, []
    while argv:
        arg = argv.pop(0)
        if arg == "--max-shards":
            max_shards = int(argv.pop(0))
        elif arg.startswith("-"):
            sys.exit("unknown flag " + arg)
        else:
            positional.append(arg)

    quant_dir = Path(positional[0] if positional else DEFAULT_QUANT_PATH)
    if len(positional) > 1:
        bf16_dir = Path(positional[1])
    else:
        stripped = quant_dir.name[: -len("-mxfp8")] if quant_dir.name.endswith("-mxfp8") else quant_dir.name
        bf16_dir = quant_dir.parent / stripped

    print("quantized = " + str(quant_dir))
    print("bf16      = " + str(bf16_dir))
    for d in (quant_dir, bf16_dir):
        if not d.is_dir():
            print("[RED] not a directory: " + str(d))
            return 1

    cfg_a, cfg_b = load_config(quant_dir), load_config(bf16_dir)
    if cfg_a is None or cfg_b is None:
        print("[RED] missing config.json on one side")
        return 1
    print_config_diff(cfg_a, cfg_b)

    print("")
    print("=== reading safetensors headers ===")
    ta = collect(quant_dir, "quantized", max_shards)
    tb = collect(bf16_dir, "bf16", max_shards)
    if ta is None or tb is None:
        return 1

    names_a, names_b = set(ta), set(tb)
    shared = names_a & names_b
    print("")
    print("=== inventory ===")
    print("  quantized tensors: " + str(len(names_a)))
    print("  bf16 tensors     : " + str(len(names_b)))
    print("  shared names     : " + str(len(shared)))

    print_pattern_summary("only in quantized", names_a - names_b, ta)
    print_pattern_summary("only in bf16", names_b - names_a, tb)

    # Shape differences on shared names, grouped by normalized name pattern.
    diffs = defaultdict(Counter)
    for name in shared:
        if ta[name][0] != tb[name][0]:
            diffs[pattern(name)][(ta[name][0], tb[name][0])] += 1
    print("")
    if diffs:
        print("=== shape mismatches on shared names ===")
        for pat, pair_counts in sorted(diffs.items(), key=lambda kv: -sum(kv[1].values())):
            for (sa, sb), count in pair_counts.most_common(3):
                numel_a, numel_b = 1, 1
                for v in sa:
                    numel_a *= v
                for v in sb:
                    numel_b *= v
                note = "same numel (pure reshape)" if numel_a == numel_b else "DIFFERENT numel"
                print("  " + pat + "   x" + str(count))
                print("      quantized " + str(list(sa)) + "   bf16 " + str(list(sb)) + "   " + note)
    else:
        print("=== shape mismatches on shared names: none ===")

    # Side-by-side view of one concrete expert, plus the hidden-size anchors.
    expert_names = sorted(n for n in shared if EXPERT_RE.search(n) and n.endswith(".weight"))
    if expert_names:
        sample = EXPERT_RE.search(expert_names[0])
        layer, expert = sample.group(1), sample.group(2)
        print("")
        print("=== layer " + layer + " / expert " + expert + " side by side ===")
        for name in expert_names:
            m = EXPERT_RE.search(name)
            if m.group(1) == layer and m.group(2) == expert:
                print("  " + name)
                print("      quantized " + str(list(ta[name][0])) + " " + ta[name][1]
                      + "   |   bf16 " + str(list(tb[name][0])) + " " + tb[name][1])

    anchors = [n for n in shared
               if n.endswith("embed_tokens.weight")
               or n.endswith("input_layernorm.weight")
               or n.endswith("o_proj.weight")]
    if anchors:
        print("")
        print("=== hidden_size anchors (non-MoE tensors) ===")
        for name in sorted(anchors)[:6]:
            print("  " + name)
            print("      quantized " + str(list(ta[name][0])) + "   |   bf16 " + str(list(tb[name][0])))

    # Verdict. Routed experts are judged on their own, because the two sides may
    # use different naming schemes (3-D fused `experts.gate_up_proj` vs per-expert
    # `experts.N.gate_proj.weight`) and then share no tensor names at all.
    hidden = cfg_b.get("hidden_size")
    moe_inter = cfg_b.get("moe_intermediate_size")
    print("")
    print("=== routed expert gate/up layout ===")
    fused_b = pick_fused_gate_up(tb)
    per_a = pick_per_expert_gate_up(ta)
    if fused_b:
        print("  bf16      " + fused_b[0] + " " + str(list(fused_b[1]))
              + "   -> per expert [" + str(fused_b[1][1]) + ", " + str(fused_b[1][2]) + "] (gate+up fused)")
    if per_a:
        print("  quantized " + per_a[0] + " " + str(list(per_a[1])) + "   (one expert, gate only)")
    expected = [moe_inter, hidden] if moe_inter and hidden else None
    if expected:
        print("  vLLM wants one expert's gate to be " + str(expected))

    expert_diffs = {p: c for p, c in diffs.items() if "experts" in p and p.endswith(".weight")}
    print("")
    if expert_diffs:
        pat = next(iter(expert_diffs))
        (sa, sb), _ = next(iter(expert_diffs[pat].most_common(1)))
        print("[RED] expert weights were RESHAPED by the quantization step.")
        print("      " + pat + ": bf16 " + str(list(sb)) + " -> quantized " + str(list(sa)) + ".")
        print("      vLLM expects [moe_intermediate_size, hidden_size] = " + str(expected)
              + ", which is what the bf16 side has.")
        return 1
    if per_a and expected and list(per_a[1]) != expected:
        rows, cols = per_a[1][0], per_a[1][1]
        print("[RED] the quantized gate tensor is " + str(list(per_a[1])) + ", not " + str(expected) + ".")
        if fused_b and [rows, cols] == [fused_b[1][1], fused_b[1][2] // 2]:
            print("      It is exactly the fused per-expert tensor " + str([fused_b[1][1], fused_b[1][2]]))
            print("      cut in half along dim 1 - the HIDDEN axis. gate and up live along dim 0")
            print("      (" + str(fused_b[1][1]) + " = 2 x " + str(moe_inter) + "), so the split axis is wrong:")
            print("        correct  : chunk(2, dim=0) -> gate [" + str(moe_inter) + ", " + str(hidden) + "]")
            print("        this file: chunk(2, dim=1) -> " + str(list(per_a[1])) + " holding half of BOTH")
            print("      Every 'gate_proj' here is really the first half of hidden for gate AND up,")
            print("      so no loader-side reshape can recover it correctly without re-pairing the")
            print("      two tensors. The quantized export is the broken artifact.")
        else:
            print("      config.json and this quantized export disagree about the expert layout.")
        return 1
    if not shared:
        print("[RED] the two directories share no tensor names; they are not the same model export.")
        return 1
    print("[GREEN] expert weights have identical shapes on both sides and match config.json.")
    print("        The load failure comes from the runtime side, not from these files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
