#!/usr/bin/env python3
"""Fast preflight for the Qwen3.8 ModelSlim metadata contract.

This intentionally does not import torch, torch_npu, vllm, or vllm_ascend, so
it can fail fast before allocating NPU memory or starting 32 worker processes.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any


SUPPORTED_MODEL_TYPES = frozenset({"qwen3_5_moe", "qwen3_5_moe_text"})
EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
MODEL_TYPE_GUARD = re.compile(r'if\s+"qwen3_5"\s+in\s+self\.config\.model_type\s*:')
MROPE_TYPE_GUARD = re.compile(
    r"isinstance\(\s*self\.rotary_emb\s*,\s*AscendMRotaryEmbedding\s*\)"
)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}, got {type(value).__name__}")
    return value


def read_packed_mapping(source_path: Path, model_types: set[str]) -> dict[str, Any]:
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (FileNotFoundError, SyntaxError) as exc:
        raise ValueError(f"cannot parse active ModelSlim source {source_path}: {exc}") from exc

    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and target.id == "packed_modules_model_mapping" and value is not None:
            if not isinstance(value, ast.Dict):
                raise ValueError("packed_modules_model_mapping is not a dictionary literal")
            # The full mapping contains a few entries that reference shared
            # constants. Evaluate literal entries independently so those
            # unrelated names do not prevent inspection of the active Qwen3.5
            # architecture alias.
            mapping: dict[str, Any] = {}
            for key_node, value_node in zip(value.keys, value.values):
                if key_node is None:
                    continue
                try:
                    key = ast.literal_eval(key_node)
                except (ValueError, TypeError):
                    continue
                if key not in model_types:
                    continue
                try:
                    mapping[key] = ast.literal_eval(value_node)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{key} packed mapping is not a literal in {source_path}") from exc
            return mapping
    raise ValueError(f"packed_modules_model_mapping not found in {source_path}")


def read_rope_patch_contract(source_path: Path) -> tuple[bool, bool, bool]:
    try:
        source = source_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"missing active Qwen3.5 patch source: {source_path}") from exc

    return (
        bool(MODEL_TYPE_GUARD.search(source)),
        bool(MROPE_TYPE_GUARD.search(source)),
        "self.rotary_emb.mrope_section" in source,
    )


def rope_contract_error(
    model_type: object,
    rope_parameters: dict[str, Any],
    patch_contract: tuple[bool, bool, bool],
) -> str | None:
    if not isinstance(model_type, str) or "qwen3_5" not in model_type:
        return None
    if "mrope_section" in rope_parameters:
        return None

    uses_model_type_guard, uses_mrope_type_guard, dereferences_mrope_section = patch_contract
    if dereferences_mrope_section and uses_model_type_guard and not uses_mrope_type_guard:
        return (
            "Qwen3.5 patch selects the M-RoPE fused path from model_type, but this model's "
            "rope_parameters has no mrope_section; runtime AscendRotaryEmbedding will raise AttributeError"
        )
    if dereferences_mrope_section and not uses_mrope_type_guard:
        return "cannot find an AscendMRotaryEmbedding type guard around the mrope_section access"
    return None


def find_layer_zero_expert_triplets(keys: set[str]) -> list[tuple[str, ...]]:
    triplets: list[tuple[str, ...]] = []
    gate_suffix = ".experts.0.gate_proj.weight"
    for key in sorted(keys):
        if ".layers.0." not in key or not key.endswith(gate_suffix):
            continue
        base = key[: -len("gate_proj.weight")]
        expected = tuple(f"{base}{projection}.weight" for projection in EXPERT_PROJECTIONS)
        if all(candidate in keys for candidate in expected):
            triplets.append(expected)
    return triplets


def check_weight_index(model_path: Path, failures: list[str]) -> None:
    index_paths = sorted(model_path.glob("*.safetensors.index.json"))
    if not index_paths:
        shard_count = sum(1 for _ in model_path.glob("*.safetensors"))
        print(f"weight_index=absent standalone_safetensors={shard_count}")
        if shard_count == 0:
            failures.append("no safetensors index and no top-level .safetensors files")
        return

    for index_path in index_paths:
        index = load_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            failures.append(f"{index_path.name} has no dictionary weight_map")
            continue
        shards = sorted({name for name in weight_map.values() if isinstance(name, str)})
        missing = [name for name in shards if not (model_path / name).is_file()]
        total_bytes = sum((model_path / name).stat().st_size for name in shards if (model_path / name).is_file())
        print(
            f"weight_index={index_path.name} tensors={len(weight_map)} "
            f"shards={len(shards)} missing_shards={len(missing)} bytes={total_bytes}"
        )
        if missing:
            failures.append(f"{index_path.name} references {len(missing)} missing shards; first={missing[0]}")


def run_check(model_path: Path, modelslim_source: Path | None, patch_source: Path | None) -> int:
    failures: list[str] = []
    print(f"model_path={model_path.resolve()}")

    try:
        config = load_json(model_path / "config.json")
    except ValueError as exc:
        print(f"RED: {exc}")
        return 1

    top_model_type = config.get("model_type")
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    text_model_type = text_config.get("model_type")
    architectures = config.get("architectures")
    effective_text_config = text_config or config
    rope_parameters = effective_text_config.get("rope_parameters")
    if not isinstance(rope_parameters, dict):
        rope_parameters = {}
    print(f"config.model_type={top_model_type!r}")
    print(f"config.text_config.model_type={text_model_type!r}")
    print(f"config.architectures={architectures!r}")
    print(f"config.quantization_config={config.get('quantization_config')!r}")
    print(f"effective_text_config.rope_parameters={json.dumps(rope_parameters, sort_keys=True)}")
    print(
        "predicted_rotary_embedding="
        + ("AscendMRotaryEmbedding" if "mrope_section" in rope_parameters else "AscendRotaryEmbedding")
    )

    observed_model_types = {
        model_type for model_type in (top_model_type, text_model_type) if model_type in SUPPORTED_MODEL_TYPES
    }
    selected_model_type = top_model_type if top_model_type in SUPPORTED_MODEL_TYPES else text_model_type
    if not observed_model_types:
        failures.append(
            "neither top-level nor text_config model_type is a supported Qwen3.5 MoE alias; "
            f"expected one of {sorted(SUPPORTED_MODEL_TYPES)!r}"
        )

    try:
        quant_description = load_json(model_path / "quant_model_description.json")
    except ValueError as exc:
        failures.append(str(exc))
        quant_description = {}

    quant_keys = set(quant_description)
    print(f"quant_description.keys={len(quant_keys)}")
    direct_key = "model.layers.0.mlp.experts.weight"
    print(f"quant_description.has_direct_failing_key={direct_key in quant_keys}")
    triplets = find_layer_zero_expert_triplets(quant_keys)
    print(f"quant_description.layer0_expert_triplets={len(triplets)}")
    if triplets:
        for key in triplets[0]:
            print(f"  expected_expert_key={key} type={quant_description[key]!r}")
    elif quant_description:
        samples = sorted(key for key in quant_keys if ".experts." in key and key.endswith(".weight"))[:6]
        for key in samples:
            print(f"  observed_expert_key={key} type={quant_description[key]!r}")
        failures.append(
            "quant_model_description.json has no complete layer-0 experts.0 "
            "gate_proj/up_proj/down_proj weight triplet"
        )

    if modelslim_source is None:
        failures.append("active vllm_ascend/quantization/modelslim_config.py was not supplied")
    else:
        print(f"active_modelslim_source={modelslim_source.resolve()}")
        try:
            packed_mapping = read_packed_mapping(modelslim_source, observed_model_types)
            qwen_mapping = packed_mapping.get(selected_model_type)
            print(f"selected_model_type={selected_model_type!r}")
            print(f"active_selected_model_mapping={qwen_mapping!r}")
            expected_experts = [f"experts.0.{projection}" for projection in EXPERT_PROJECTIONS]
            if not isinstance(qwen_mapping, dict) or qwen_mapping.get("experts") != expected_experts:
                failures.append(
                    "active ModelSlim source lacks the required "
                    f"packed_modules_model_mapping[{selected_model_type!r}]['experts'] mapping"
                )
        except ValueError as exc:
            failures.append(str(exc))

    if patch_source is None:
        failures.append("active vllm_ascend/patch/worker/patch_qwen3_5.py was not supplied")
    else:
        print(f"active_qwen3_5_patch_source={patch_source.resolve()}")
        try:
            patch_contract = read_rope_patch_contract(patch_source)
            print(f"patch.uses_model_type_guard={patch_contract[0]}")
            print(f"patch.uses_mrope_type_guard={patch_contract[1]}")
            print(f"patch.dereferences_mrope_section={patch_contract[2]}")
            error = rope_contract_error(selected_model_type, rope_parameters, patch_contract)
            if error is not None:
                failures.append(error)
        except ValueError as exc:
            failures.append(str(exc))

    try:
        check_weight_index(model_path, failures)
    except ValueError as exc:
        failures.append(str(exc))

    if failures:
        print("VERDICT=RED")
        for failure in failures:
            print(f"RED: {failure}")
        return 1

    print("VERDICT=GREEN")
    print("The static ModelSlim contract is present; next inspect the runtime model_type and imported checkout.")
    return 0


def self_test() -> int:
    prefix = "model.layers.0.mlp.experts.0."
    good = {f"{prefix}{projection}.weight" for projection in EXPERT_PROJECTIONS}
    bad = good - {f"{prefix}down_proj.weight"}
    assert len(find_layer_zero_expert_triplets(good)) == 1
    assert find_layer_zero_expert_triplets(bad) == []

    regular_rope = {"rope_type": "default", "rope_theta": 10_000_000}
    mrope = {**regular_rope, "mrope_section": [11, 11, 10]}
    unsafe_patch = (True, False, True)
    guarded_patch = (False, True, True)
    unsafe_error = rope_contract_error("qwen3_5_moe_text", regular_rope, unsafe_patch)
    guarded_error = rope_contract_error("qwen3_5_moe_text", regular_rope, guarded_patch)
    assert unsafe_error is not None
    assert guarded_error is None
    assert rope_contract_error("qwen3_5_moe_text", mrope, unsafe_patch) is None
    print(f"SELF_TEST_UNSAFE=RED reason={unsafe_error}")
    print("SELF_TEST_GUARDED=GREEN")
    print("SELF_TEST=PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs="?", type=Path)
    parser.add_argument("--modelslim-source", type=Path)
    parser.add_argument("--patch-source", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and args.model_path is None:
        parser.error("model_path is required unless --self-test is used")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    return run_check(args.model_path, args.modelslim_source, args.patch_source)


if __name__ == "__main__":
    sys.exit(main())
