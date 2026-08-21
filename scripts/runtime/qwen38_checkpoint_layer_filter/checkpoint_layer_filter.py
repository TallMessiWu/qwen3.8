"""Skip truncated Qwen3.8 language-layer tensors before reading them."""

import functools
import importlib.abc
import importlib.machinery
import os
import re
import sys
from types import ModuleType
from typing import Any


ENV_NAME = "QWEN38_CHECKPOINT_LAYER_LIMIT"
TARGET_MODULE = "vllm.model_executor.model_loader.weight_utils"
PATCH_MARKER = "_qwen38_checkpoint_layer_filter_limit"
LANGUAGE_LAYER_PATTERN = re.compile(
    r"^(?:model\.)?(?:language_model\.)?(?:model\.)?layers\.(\d+)\."
)


def should_skip_for_layer_limit(name: str, layer_limit: int) -> bool:
    match = LANGUAGE_LAYER_PATTERN.match(name)
    return match is not None and int(match.group(1)) >= layer_limit


def patch_weight_utils(
    module: ModuleType | Any,
    layer_limit: int,
    *,
    announce: bool = True,
) -> None:
    current_limit = getattr(module, PATCH_MARKER, None)
    if current_limit is not None:
        if current_limit != layer_limit:
            raise RuntimeError(
                f"checkpoint layer filter already uses {current_limit}, "
                f"cannot change it to {layer_limit}"
            )
        return

    original_filter = module.should_skip_weight
    first_skipped = True

    @functools.wraps(original_filter)
    def filtered(name: str, local_expert_ids: set[int] | None) -> bool:
        nonlocal first_skipped
        if should_skip_for_layer_limit(name, layer_limit):
            if announce and first_skipped:
                print(
                    "QWEN38_CHECKPOINT_LAYER_FILTER=GREEN "
                    f"first_skipped={name} layer_limit={layer_limit}",
                    flush=True,
                )
                first_skipped = False
            return True
        return original_filter(name, local_expert_ids)

    module.should_skip_weight = filtered
    setattr(module, PATCH_MARKER, layer_limit)
    if announce:
        print(
            "QWEN38_CHECKPOINT_LAYER_FILTER=installed "
            f"layer_limit={layer_limit} stage=before_get_tensor",
            flush=True,
        )


class _WeightUtilsLoader(importlib.abc.Loader):
    def __init__(self, wrapped: Any, layer_limit: int):
        self.wrapped = wrapped
        self.layer_limit = layer_limit

    def create_module(self, spec):
        create_module = getattr(self.wrapped, "create_module", None)
        return create_module(spec) if create_module is not None else None

    def exec_module(self, module) -> None:
        self.wrapped.exec_module(module)
        patch_weight_utils(module, self.layer_limit)


class _WeightUtilsFinder(importlib.abc.MetaPathFinder):
    def __init__(self, layer_limit: int):
        self.layer_limit = layer_limit

    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET_MODULE:
            return None

        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _WeightUtilsLoader(spec.loader, self.layer_limit)
        return spec


def install_from_env() -> None:
    raw_limit = os.environ.get(ENV_NAME)
    if raw_limit is None:
        return
    try:
        layer_limit = int(raw_limit)
    except ValueError as exc:
        raise RuntimeError(f"{ENV_NAME} must be a positive integer") from exc
    if layer_limit <= 0:
        raise RuntimeError(f"{ENV_NAME} must be a positive integer")

    loaded_module = sys.modules.get(TARGET_MODULE)
    if loaded_module is not None:
        patch_weight_utils(loaded_module, layer_limit)
        return

    for finder in sys.meta_path:
        if isinstance(finder, _WeightUtilsFinder):
            if finder.layer_limit != layer_limit:
                raise RuntimeError(
                    f"checkpoint layer filter already uses {finder.layer_limit}, "
                    f"cannot change it to {layer_limit}"
                )
            return
    sys.meta_path.insert(0, _WeightUtilsFinder(layer_limit))
