#!/usr/bin/env python3
"""Verify a ModelSlim quant description against the qwen3_5_moe_text mapping fix.

Context: loading the mxfp8 2.4T checkpoint failed with
KeyError('model.layers.0.mlp.experts.weight') because
vllm_ascend/quantization/modelslim_config.py had no packed-module mapping for
model_type "qwen3_5_moe_text" (fix mirrors vllm-ascend PR #14238).

This script re-runs, offline, the exact lookup vLLM Ascend performs at load
time with the fixed mapping table:
  1. config.json model_type must be covered by the mapping table.
  2. Every ".weight" key in quant_model_description.json must resolve through
     the packed-shard lookup (all shards present, consistent quant type;
     FLOAT counts as a valid "skip quantization").
  3. Required lookups (embed_tokens, lm_head, layer-0 MoE modules) must
     succeed even if the description omitted them entirely.
  4. Expert coverage per layer must match config.json (num_experts and
     num_hidden_layers).

Pure stdlib, read-only, no torch/vllm import, never touches the NPU.
Usage: python3 check_quant_desc_qwen35_moe_text.py [MODEL_PATH]
Final line is [GREEN] (exit 0) or [RED] (exit 1).
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_MODEL_PATH = "/mnt/share/weight/Qwen3.8-2.4T-A95B-mxfp8"

# Mirrors packed_modules_model_mapping["qwen3_5_moe_text"] after the fix.
PACKED = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
    "experts": ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
}
MAPPED_MODEL_TYPES = {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}

# register_scheme() snapshot from vllm_ascend/quantization/methods/ (taken from
# the since-deleted junlin bugfix branch; the mapping now lives on upstream/main).
MOE_SCHEMES = {
    "W4A16", "W4A16_MXFP4", "W4A4_MXFP4", "W4A8_DYNAMIC", "W4A8_MXFP",
    "W8A8FP8_DYNAMIC", "W8A8_DYNAMIC", "W8A8_MXFP8",
}

SHARD_TO_FUSED = {}
for fused, shards in PACKED.items():
    if fused == "experts":
        continue
    for shard in shards:
        SHARD_TO_FUSED[shard] = fused

EXPERT_SHARDS = ("gate_proj", "up_proj", "down_proj")


def fmt_value(value):
    """Render a description value compactly; dict/list values are JSON-encoded."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)[:160]


def key_pattern(key):
    return ".".join("*" if part.isdigit() else part for part in key.split("."))


def reduce_to_query_prefix(weight_key):
    """Map one description key to the module prefix vLLM queries for it."""
    name = weight_key[: -len(".weight")]
    parts = name.split(".")
    if (
        len(parts) >= 3
        and parts[-3] == "experts"
        and parts[-2].isdigit()
        and parts[-1] in EXPERT_SHARDS
    ):
        return ".".join(parts[:-2])
    if parts[-1] in SHARD_TO_FUSED:
        return ".".join(parts[:-1] + [SHARD_TO_FUSED[parts[-1]]])
    return name


def lookup(desc, prefix):
    """Replicates get_linear_quant_type() with the fixed mapping.

    Returns (status, detail): status in {"quant", "float", "missing", "mixed"}.
    """
    proj_name = prefix.split(".")[-1]
    if proj_name in PACKED:
        quant_type = None
        for shard_proj_name in PACKED[proj_name]:
            shard_key = prefix.replace(proj_name, shard_proj_name) + ".weight"
            if shard_key not in desc:
                return "missing", shard_key
            shard_type = desc[shard_key]
            if quant_type is None:
                quant_type = shard_type
            elif shard_type != quant_type:
                return "mixed", f"{shard_key}={shard_type} vs {quant_type}"
        return ("float", quant_type) if quant_type == "FLOAT" else ("quant", quant_type)
    key = prefix + ".weight"
    if key not in desc:
        return "missing", key
    value = desc[key]
    return ("float", value) if value == "FLOAT" else ("quant", value)


QUANT_DTYPES = {"F8_E4M3", "F8_E5M2", "F8_E8M0", "I8", "U8", "I4", "U4", "F4_E2M1"}


def read_safetensors_header(path):
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(header_len))


def check_safetensors(model_path, problems, warnings):
    """Inspect checkpoint tensor names/dtypes without loading tensor data.

    Returns True if the safetensors look quantized, False if plain bf16/fp16,
    None if no safetensors were found.
    """
    index_path = model_path / "model.safetensors.index.json"
    tensor_files = {}
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        for name, filename in weight_map.items():
            tensor_files[name] = model_path / filename
    else:
        shards = sorted(model_path.glob("*.safetensors"))
        if not shards:
            warnings.append("no *.safetensors found; skipped checkpoint reality check")
            return None
        print(f"\nno index.json; scanning {len(shards)} safetensors headers")
        for shard in shards:
            for name in read_safetensors_header(shard):
                if name != "__metadata__":
                    tensor_files[name] = shard

    print(f"\nsafetensors: {len(tensor_files)} tensors")
    name_patterns = Counter(key_pattern(name) for name in tensor_files)
    scale_like = [n for n in tensor_files if "scale" in n.lower() or "offset" in n.lower()]
    print(f"  {len(name_patterns)} tensor-name patterns, {len(scale_like)} scale/offset tensors")
    for pattern, count in sorted(name_patterns.items()):
        print(f"    {count:6d}  {pattern}")

    focus = {}
    for name in tensor_files:
        for tag, needle in (
            ("experts.gate_up_proj", ".mlp.experts.gate_up_proj"),
            ("experts.down_proj", ".mlp.experts.down_proj"),
            ("in_proj_qkv", ".linear_attn.in_proj_qkv"),
            ("self_attn.q_proj", ".self_attn.q_proj"),
            ("embed_tokens", "embed_tokens"),
        ):
            if needle in name and tag not in focus:
                focus[tag] = name
    for name in scale_like[:3]:
        focus[f"scale:{key_pattern(name)}"] = name

    sample_dtypes = set()
    headers = {}
    print("  representative tensors:")
    for tag, name in sorted(focus.items()):
        shard = tensor_files[name]
        if shard not in headers:
            headers[shard] = read_safetensors_header(shard)
        info = headers[shard].get(name)
        if info is None:
            warnings.append(f"{name} listed in index but absent from {shard.name}")
            continue
        print(f"    {tag:28s} {name} dtype={info['dtype']} shape={info['shape']}")
        sample_dtypes.add(info["dtype"])

    quantized = bool(sample_dtypes & QUANT_DTYPES) or bool(scale_like)
    print(f"  sample dtypes={sorted(sample_dtypes)} -> checkpoint looks {'QUANTIZED' if quantized else 'UNQUANTIZED'}")
    return quantized


def main():
    model_path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH)
    problems = []
    warnings = []

    config = json.loads((model_path / "config.json").read_text())
    text_config = config.get("text_config") or {}
    model_type = config.get("model_type")
    print(f"config.json model_type={model_type!r} architectures={config.get('architectures')}")
    if text_config:
        print(f"  nested text_config.model_type={text_config.get('model_type')!r}")
    num_layers = config.get("num_hidden_layers") or text_config.get("num_hidden_layers")
    num_experts = config.get("num_experts") or text_config.get("num_experts")
    print(f"  num_hidden_layers={num_layers} num_experts={num_experts}")
    if model_type not in MAPPED_MODEL_TYPES:
        problems.append(
            f"model_type {model_type!r} is NOT in packed_modules_model_mapping "
            f"({sorted(MAPPED_MODEL_TYPES)}) — the fix does not cover this checkpoint"
        )

    desc = json.loads((model_path / "quant_model_description.json").read_text())
    weight_keys = [k for k in desc if k.endswith(".weight")]
    print(f"\nquant_model_description.json: {len(desc)} keys, {len(weight_keys)} '.weight' keys")
    value_kinds = Counter(v if isinstance(v, str) else f"<{type(v).__name__}>" for v in desc.values())
    print(f"  value distribution: {dict(value_kinds)}")
    non_str = {k: v for k, v in desc.items() if not isinstance(v, str)}
    if non_str:
        non_str_weight = [k for k in non_str if k.endswith(".weight")]
        print(
            f"  {len(non_str)} keys carry non-string values "
            f"({len(non_str_weight)} of them '.weight' keys); one sample per key pattern:"
        )
        shown = set()
        for key, value in non_str.items():
            pattern = key_pattern(key)
            if pattern in shown:
                continue
            shown.add(pattern)
            print(f"    {key} = {fmt_value(value)}")
            if len(shown) >= 15:
                print(f"    ... ({len(non_str)} non-string keys total)")
                break
        if non_str_weight:
            warnings.append(
                f"{len(non_str_weight)} '.weight' keys have dict/list values; "
                "modelslim_config compares values against plain strings"
            )

    patterns = Counter()
    for key in desc:
        patterns[key_pattern(key)] += 1
    print(f"  {len(patterns)} distinct key patterns:")
    for pattern, count in sorted(patterns.items()):
        print(f"    {count:6d}  {pattern}")

    # Reverse pass: every description weight key must resolve via the fixed lookup.
    prefixes = sorted({reduce_to_query_prefix(k) for k in weight_keys})
    by_status = defaultdict(list)
    quant_types_seen = Counter()
    for prefix in prefixes:
        status, detail = lookup(desc, prefix)
        by_status[status].append((prefix, detail))
        if status == "quant":
            quant_types_seen[fmt_value(detail)] += 1
    print(
        f"\nlookup over {len(prefixes)} module prefixes: "
        f"{len(by_status['quant'])} quantized, {len(by_status['float'])} FLOAT-skipped, "
        f"{len(by_status['missing'])} missing shards, {len(by_status['mixed'])} mixed types"
    )
    print(f"  quant types resolved: {dict(quant_types_seen)}")
    for status in ("missing", "mixed"):
        for prefix, detail in by_status[status][:20]:
            problems.append(f"{status}: prefix {prefix!r} -> {detail}")
        if len(by_status[status]) > 20:
            problems.append(f"... {len(by_status[status]) - 20} more {status} prefixes")

    non_model = [p for p in prefixes if not p.startswith("model.") and p != "lm_head"]
    if non_model:
        print(f"  note: {len(non_model)} prefixes outside model./lm_head (MTP etc.):")
        for prefix in non_model[:10]:
            status, detail = lookup(desc, prefix)
            print(f"    {prefix}  -> {status}: {fmt_value(detail)}")

    # Forward pass: lookups the model performs even if the description omits them.
    required = ["model.embed_tokens", "lm_head", "model.layers.0.mlp.experts",
                "model.layers.0.mlp.gate"]
    for candidate in ("model.layers.0.mlp.shared_expert.gate_up_proj",
                      "model.layers.0.mlp.shared_expert.down_proj"):
        if any(k.startswith("model.layers.0.mlp.shared_expert.") for k in desc):
            required.append(candidate)
    print("\nrequired lookups:")
    for prefix in required:
        status, detail = lookup(desc, prefix)
        print(f"  {prefix:55s} -> {status}: {fmt_value(detail)}")
        if status in ("missing", "mixed"):
            problems.append(f"required lookup failed: {prefix} -> {status} ({fmt_value(detail)})")
    status, detail = lookup(desc, "model.layers.0.mlp.experts")
    if status == "quant":
        if not isinstance(detail, str):
            problems.append(
                f"experts quant value is a {type(detail).__name__} ({fmt_value(detail)}); "
                "modelslim_config expects a plain string quant type -- schema adaptation needed"
            )
        elif detail not in MOE_SCHEMES:
            problems.append(
                f"experts quant type {detail!r} has no registered 'moe' scheme {sorted(MOE_SCHEMES)}"
            )

    # Expert coverage: every layer should describe the same, full expert set.
    experts_per_layer = defaultdict(set)
    for key in weight_keys:
        parts = key.split(".")
        if key.startswith("model.layers.") and "experts" in parts:
            idx = parts.index("experts")
            if parts[2].isdigit() and idx + 1 < len(parts) and parts[idx + 1].isdigit():
                experts_per_layer[int(parts[2])].add(int(parts[idx + 1]))
    if experts_per_layer:
        counts = {layer: len(ids) for layer, ids in experts_per_layer.items()}
        distinct = Counter(counts.values())
        print(f"\nexpert coverage: {len(counts)} MoE layers, experts-per-layer counts {dict(distinct)}")
        if num_layers and len(counts) != num_layers:
            warnings.append(f"{len(counts)} MoE layers described vs num_hidden_layers={num_layers}")
        if num_experts and set(distinct) != {num_experts}:
            problems.append(f"expert count per layer {dict(distinct)} != config num_experts={num_experts}")
    else:
        fused_expert_keys = [
            k for k in desc if ".mlp.experts.gate_up_proj" in k or ".mlp.experts.down_proj" in k
        ]
        if fused_expert_keys:
            problems.append(
                f"description keys use the transformers-5.x fused expert layout "
                f"({len(fused_expert_keys)} keys like {fused_expert_keys[0]!r}, no '.weight' suffix); "
                "modelslim_config only understands per-expert keys (experts.0.gate_proj.weight) "
                "-- needs code adaptation or description conversion"
            )
        else:
            problems.append("no per-expert keys (model.layers.N.mlp.experts.M.*) found at all")

    quantized = check_safetensors(model_path, problems, warnings)
    all_float = not quant_types_seen
    if quantized is True and all_float:
        problems.append(
            "safetensors carry quantized tensors but every description value is FLOAT "
            "-- quant_model_description.json does not describe this checkpoint's quantization"
        )
    elif quantized is False:
        problems.append(
            "safetensors are plain bf16/fp16 -- the checkpoint is not actually quantized; "
            "serve it without --quantization ascend or fetch the real mxfp8 weights"
        )

    for warning in warnings:
        print(f"[WARN] {warning}")
    if problems:
        print(f"\n[RED] {len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\n[GREEN] quant description resolves cleanly with the qwen3_5_moe_text mapping fix")
    return 0


if __name__ == "__main__":
    sys.exit(main())
