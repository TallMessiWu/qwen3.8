#!/usr/bin/env python3
"""Fast preflight for the Qwen3.8 ModelSlim metadata contract.

This intentionally does not import torch, torch_npu, vllm, or vllm_ascend, so
it can fail fast before allocating NPU memory or starting 32 worker processes.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any


EXPECTED_MODEL_TYPE = "qwen3_5_moe"
EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


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


def read_packed_mapping(source_path: Path) -> dict[str, Any]:
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
            # unrelated names do not prevent inspection of qwen3_5_moe.
            mapping: dict[str, Any] = {}
            for key_node, value_node in zip(value.keys, value.values):
                if key_node is None:
                    continue
                try:
                    key = ast.literal_eval(key_node)
                except (ValueError, TypeError):
                    continue
                if key != EXPECTED_MODEL_TYPE:
                    continue
                try:
                    mapping[key] = ast.literal_eval(value_node)
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"{EXPECTED_MODEL_TYPE} packed mapping is not a literal in {source_path}"
                    ) from exc
            return mapping
    raise ValueError(f"packed_modules_model_mapping not found in {source_path}")


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


def run_check(model_path: Path, modelslim_source: Path | None) -> int:
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
    print(f"config.model_type={top_model_type!r}")
    print(f"config.text_config.model_type={text_model_type!r}")
    print(f"config.architectures={architectures!r}")
    print(f"config.quantization_config={config.get('quantization_config')!r}")

    if EXPECTED_MODEL_TYPE not in {top_model_type, text_model_type}:
        failures.append(
            f"neither top-level nor text_config model_type is {EXPECTED_MODEL_TYPE!r}; "
            "the Qwen3.5 MoE packed mapping may not be selected"
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
            packed_mapping = read_packed_mapping(modelslim_source)
            qwen_mapping = packed_mapping.get(EXPECTED_MODEL_TYPE)
            print(f"active_qwen3_5_moe_mapping={qwen_mapping!r}")
            expected_experts = [f"experts.0.{projection}" for projection in EXPERT_PROJECTIONS]
            if not isinstance(qwen_mapping, dict) or qwen_mapping.get("experts") != expected_experts:
                failures.append(
                    "active ModelSlim source lacks the required "
                    f"packed_modules_model_mapping[{EXPECTED_MODEL_TYPE!r}]['experts'] mapping"
                )
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
    print("SELF_TEST=PASS")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs="?", type=Path)
    parser.add_argument("--modelslim-source", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and args.model_path is None:
        parser.error("model_path is required unless --self-test is used")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    return run_check(args.model_path, args.modelslim_source)


if __name__ == "__main__":
    sys.exit(main())
