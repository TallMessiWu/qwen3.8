#!/usr/bin/env python3
"""Startup-phase smoke test: REAL vllm + REAL vllm_ascend imports, no NPU.

The e2e simulation mocks vllm away, which is exactly why an import-path typo
(`vllm.attention.backends.abstract`) survived to the server. This test closes
that hole: only torch_npu (physically absent) is replaced by a permissive
dummy; every vllm module resolves against the real submodule checkout and
every vllm_ascend module against the real worktree, so the exact path the
engine takes at startup -- resolve_obj_by_qualname -> import attention_qfa ->
class surface -- runs for real.

Heavy package __init__ files are bypassed with the namespace-shell trick
(a bare module carrying only __path__), so submodule imports execute the real
files without dragging in LLM entrypoints.

Run:  python scripts/tests/test_qfa_backend_import_smoke.py
      QFA_WORKTREE=junlin-qfa-m3 python scripts/tests/test_qfa_backend_import_smoke.py
"""

import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

if not sys.flags.utf8_mode:
    # torch inductor templates are UTF-8; Windows defaults to GBK
    raise SystemExit(subprocess.run([sys.executable, "-X", "utf8", *sys.argv]).returncode)

REPO = Path(__file__).resolve().parents[2]
WORKTREE = os.environ.get("QFA_WORKTREE", "junlin-qfa")
VLLM_SRC = REPO / "vllm"
ASCEND_SRC = REPO / "vllm-ascend" / WORKTREE


class _DummyMeta(type):
    """Metaclass making every dummy attribute a genuine CLASS: usable in
    isinstance() tuples (torch.flop_counter does that with triton's
    JITFunction), inheritable, callable, and `X | None`-annotatable."""

    def __getattr__(cls, item):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)  # keep inspect/copy protocols honest
        return _make_dummy(f"{cls.__qualname__}.{item}")

    def __or__(cls, other):  # `torch.npu.NPUGraph | None` annotations
        return object

    __ror__ = __or__

    def __repr__(cls):
        return f"<dummy class {cls.__qualname__}>"


def _instance_getattr(self, item):
    if item.startswith("__") and item.endswith("__"):
        raise AttributeError(item)
    return _make_dummy(f"{type(self).__qualname__}.{item}")


def _make_dummy(name):
    return _DummyMeta(name.rsplit(".", 1)[-1], (object,), {
        "__qualname__": name,
        "__getattr__": _instance_getattr,
        "__call__": lambda self, *a, **k: _make_dummy(f"{type(self).__qualname__}()")(),
        "__getitem__": lambda self, key: _make_dummy(f"{type(self).__qualname__}[]")(),
        # without __iter__, the legacy iteration protocol would pull
        # __getitem__[0..] forever (a real hang we hit); iterate as empty
        "__iter__": lambda self: iter(()),
        "__len__": lambda self: 0,
        "__repr__": lambda self: f"<dummy {type(self).__qualname__}>",
        "__init__": lambda self, *a, **k: None,
    })


class _Dummy:
    """Instance-flavored permissive stand-in (for torch.npu attribute)."""

    def __init__(self, name="dummy"):
        self._name = name

    def __getattr__(self, item):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        return _make_dummy(f"{self._name}.{item}")

    def __repr__(self):
        return f"<dummy {self._name}>"


def _dummy_module(name):
    import importlib.machinery

    mod = types.ModuleType(name)

    def _getattr(item, _n=name):
        if item == "__version__":
            return "0.0.0+dummy"  # metadata, not a protocol dunder
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        return _make_dummy(f"{_n}.{item}")

    mod.__getattr__ = _getattr
    mod.__path__ = []  # make it a package so submodule imports also resolve
    # a real ModuleSpec keeps importlib.util.find_spec() probes working
    # (transformers' is_torch_npu_available does exactly that)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return mod


def _shell_package(name, path):
    """Register a namespace shell so `import name.sub` executes the real
    sub-module files under `path` without running name/__init__.py."""
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    sys.modules[name] = mod
    return mod


class _DummyFinder:
    """meta_path fallback: any module under the given prefixes (deps that are
    physically absent on this machine but present on the server) resolves to
    a permissive dummy. Registered LAST, so real packages always win."""

    def __init__(self, prefixes):
        self.prefixes = prefixes

    def find_module(self, fullname, path=None):  # pragma: no cover - py<3.12
        return self if self._match(fullname) else None

    def find_spec(self, fullname, path=None, target=None):
        import importlib.machinery

        if not self._match(fullname):
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def _match(self, fullname):
        return any(fullname == p or fullname.startswith(p + ".") for p in self.prefixes)

    def create_module(self, spec):
        return _dummy_module(spec.name)

    def exec_module(self, module):
        pass


def check(name, ok, detail=""):
    print(f"  [{name}] {'GREEN' if ok else 'RED'}{' ' + detail if detail else ''}")
    return ok


def main() -> int:
    assert VLLM_SRC.is_dir() and ASCEND_SRC.is_dir(), (VLLM_SRC, ASCEND_SRC)
    ok_all = True

    # physically absent server deps resolve to permissive dummies: the NPU
    # stack plus unix-only bits this Windows box cannot install. NOTE: global
    # `triton` is deliberately NOT dummied - torch inductor probes it and its
    # deep type contracts reject stand-ins; letting torch see "no triton"
    # (the true state of this box) keeps it on its native fallback, while the
    # vllm_ascend triton-kernel subtree is dummied at its own package path.
    sys.meta_path.append(_DummyFinder((
        "torch_npu", "torchair", "uvloop", "vllm_ascend.ops.triton")))
    # torch.npu is normally installed by torch_npu; provide the same dummy
    import torch

    if not hasattr(torch, "npu"):
        torch.npu = _Dummy("torch.npu")
    # symbols torch_npu injects into torch.distributed at import time
    if not hasattr(torch.distributed, "is_hccl_available"):
        torch.distributed.is_hccl_available = lambda: False

    # vllm: real modules behind a namespace shell (its package __init__ pulls
    # the full LLM entrypoint stack, far beyond what startup needs here).
    # vllm_ascend: REAL package import including its real __init__.py, so the
    # module-initialization order matches the server exactly.
    vllm_shell = _shell_package("vllm", VLLM_SRC / "vllm")
    # normally set by vllm/__init__.py (skipped by the shell); must match the
    # checked-out tag - vllm_ascend's patch system branches on it
    vllm_shell.__version__ = "0.27.1"
    # top-level re-exports vllm/__init__.py would provide, resolved lazily
    # from their real defining modules
    _vllm_top = {"ModelRegistry": "vllm.model_executor.models"}

    def _vllm_getattr(item):
        if item in _vllm_top:
            return getattr(importlib.import_module(_vllm_top[item]), item)
        raise AttributeError(item)

    vllm_shell.__getattr__ = _vllm_getattr
    sys.path.insert(0, str(ASCEND_SRC))

    # _build_info is generated by setup.py at install time (absent in the
    # source tree); pre-register the A5 flavor the server build produces.
    build_info = types.ModuleType("vllm_ascend._build_info")
    build_info.__device_type__ = "A5"
    sys.modules["vllm_ascend._build_info"] = build_info

    # ---- 1) the exact import that broke on the server ----
    try:
        backend_mod = importlib.import_module("vllm.v1.attention.backend")
        ok = hasattr(backend_mod, "AttentionType")
        ok_all &= check("VLLM-BACKEND-MODULE", ok)
    except Exception as e:  # noqa: BLE001
        ok_all &= check("VLLM-BACKEND-MODULE", False, f"{type(e).__name__}: {e}")
        print("    (cannot continue without the real vllm attention module)")
        return 1

    # ---- 1.5) replay the REAL plugin startup order ----
    # vllm loads entry points in this order before any backend resolve:
    # platform_plugins (vllm_ascend:register), then the general_plugins
    # (connector / model_loader / service_profiling / register_model - the
    # last one imports models+ops, which is what makes device_op's module
    # cycle never fire on the server). Cold-importing attention_qfa without
    # this order hits init cycles real startup never sees.
    try:
        vllm_ascend_pkg = importlib.import_module("vllm_ascend")
        platform_path = vllm_ascend_pkg.register()
        for fn in ("register_connector", "register_model_loader",
                   "register_service_profiling", "register_model"):
            getattr(vllm_ascend_pkg, fn)()
        importlib.import_module(platform_path.rsplit(".", 1)[0]
                                if "." in platform_path else "vllm_ascend.platform")
        ok_all &= check("PLUGIN-STARTUP-ORDER", True, f"platform={platform_path}")
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        ok_all &= check("PLUGIN-STARTUP-ORDER", False, f"{type(e).__name__}: {e}")
        return 1

    # ---- 2) resolve the QFA backend exactly like the engine selector ----
    try:
        from vllm.utils.import_utils import resolve_obj_by_qualname

        backend_cls = resolve_obj_by_qualname(
            "vllm_ascend.attention.attention_qfa.AscendQfaAttentionBackend")
        ok_all &= check("RESOLVE-QFA-BACKEND", True)
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        ok_all &= check("RESOLVE-QFA-BACKEND", False, f"{type(e).__name__}: {e}")
        return 1

    # ---- 3) startup-time class surface, as the model runner uses it ----
    # (inside a vllm config context, exactly like the running engine)
    try:
        from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

        cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
        with set_current_vllm_config(cfg):
            impl_cls = backend_cls.get_impl_cls()
            builder_cls = backend_cls.get_builder_cls()
        shape = backend_cls.get_kv_cache_shape(100, 128, 4, 256)
        ok = impl_cls.__name__ == "AscendQfaAttentionBackendImpl"
        ok &= builder_cls is not None
        ok &= shape[2] == 128 and shape[3] == 4 and shape[4] == 256 + 4
        ok_all &= check("BACKEND-SURFACE", ok, f"shape={shape}")
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        ok_all &= check("BACKEND-SURFACE", False, f"{type(e).__name__}: {e}")

    # ---- 3b) the enum lookup vllm's Attention.__init__ does on get_name() ----
    # vllm 0.27 resolves AttentionBackendEnum[backend.get_name()] while building
    # every attention layer. The enum is closed; third-party backends must answer
    # with the registered "CUSTOM" placeholder (see attention_v1's
    # register_backend call) or the engine dies with "Unknown attention backend"
    # before the first layer exists.
    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        name = backend_cls.get_name()
        AttentionBackendEnum[name]  # ValueError if not an enum member
        ok_all &= check("BACKEND-ENUM-NAME", True, f"get_name()={name!r}")
    except Exception as e:  # noqa: BLE001
        ok_all &= check("BACKEND-ENUM-NAME", False, f"{type(e).__name__}: {e}")

    # ---- 3c) the KV pool budget must match the shape the backend carves ----
    # The engine sizes the cache pool from AscendQfaAttentionSpec.page_size_bytes
    # while the backend views it through get_kv_cache_shape. If those two drift
    # apart the pool comes up short and initialize_kv_cache dies on a view()
    # ("shape is invalid for input of size ..."), so pin them to each other.
    try:
        import torch
        from vllm_ascend.core.kv_cache_interface import AscendQfaAttentionSpec

        bs, nkv, d = 128, 4, 256
        spec = AscendQfaAttentionSpec(
            block_size=bs, num_kv_heads=nkv, head_size=d, dtype=torch.bfloat16)
        shape = backend_cls.get_kv_cache_shape(1, bs, nkv, d)
        shape_bytes = 1
        for dim in shape:
            shape_bytes *= dim
        shape_bytes *= torch.finfo(torch.bfloat16).bits // 8
        ok = spec.page_size_bytes == shape_bytes
        # and it must exceed the plain K/V pair by exactly the two scale planes
        ok &= spec.page_size_bytes - 2 * bs * nkv * d * 2 == 2 * bs * nkv * (d // 64) * 2
        # Hybrid models stretch the scheduler block to match the Mamba page, so
        # the cache is laid out in kernel blocks instead. The operator reads
        # block_size off the K plane and MXFP8 only accepts 64/128/.../1024, so
        # the kernel block must stay legal and the stretched page must still
        # divide into exactly `ratio` kernel blocks (no bytes gained or lost).
        kernel_bs = backend_cls.get_supported_kernel_block_sizes()[0]
        ratio = 12  # what unify_kv_cache_specs picks for 27B + GDN
        stretched = AscendQfaAttentionSpec(
            block_size=kernel_bs * ratio, num_kv_heads=nkv, head_size=d,
            dtype=torch.bfloat16)
        ok &= kernel_bs in (64, 128, 256, 512, 1024) and kernel_bs % 64 == 0
        ok &= stretched.page_size_bytes == ratio * shape_bytes
        ok_all &= check("SPEC-PAGE-BUDGET", ok,
                        f"page={spec.page_size_bytes} shape={shape}={shape_bytes}B "
                        f"kernel_bs={kernel_bs} stretched={stretched.page_size_bytes}")
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        ok_all &= check("SPEC-PAGE-BUDGET", False, f"{type(e).__name__}: {e}")

    # ---- 3d) the spec swap the model runner performs on QFA layers ----
    # get_kv_cache_spec() rebuilds the layer's FullAttentionSpec as the QFA one
    # by copying every init field; the registry must then still resolve it to a
    # full-attention group, or KV cache grouping asserts at startup.
    try:
        from dataclasses import fields

        from vllm.v1.kv_cache_interface import FullAttentionSpec
        from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
        from vllm_ascend.platform import NPUPlatform

        plain = FullAttentionSpec(
            block_size=bs, num_kv_heads=nkv, head_size=d, dtype=torch.bfloat16)
        swapped = AscendQfaAttentionSpec(
            **{f.name: getattr(plain, f.name) for f in fields(plain) if f.init})
        ok = swapped.page_size_bytes > plain.page_size_bytes
        with set_current_vllm_config(cfg):
            NPUPlatform.register_custom_kv_cache_specs(cfg)
            ok &= KVCacheSpecRegistry.get_uniform_type_base_spec(swapped) is FullAttentionSpec
        ok_all &= check("SPEC-SWAP", ok,
                        f"{plain.page_size_bytes} -> {swapped.page_size_bytes} bytes/page")
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        ok_all &= check("SPEC-SWAP", False, f"{type(e).__name__}: {e}")

    # ---- 4) envs + platform selector branch (the QFA env gate) ----
    try:
        os.environ["VLLM_ASCEND_ENABLE_QFA"] = "1"
        envs_mod = importlib.import_module("vllm_ascend.envs")
        ok = envs_mod.VLLM_ASCEND_ENABLE_QFA is True
        ok_all &= check("ENVS-GATE", ok)
    except Exception as e:  # noqa: BLE001
        ok_all &= check("ENVS-GATE", False, f"{type(e).__name__}: {e}")

    # ---- 5) every module the QFA file pulls in, imported for real ----
    try:
        qfa_mod = importlib.import_module("vllm_ascend.attention.attention_qfa")
        parent = importlib.import_module("vllm_ascend.attention.attention_v1")
        ok = issubclass(qfa_mod.AscendQfaAttentionBackendImpl,
                        parent.AscendAttentionBackendImpl)
        # methods the engine calls must exist on the real class hierarchy
        for method in ("forward", "forward_impl", "reshape_and_cache"):
            ok &= callable(getattr(qfa_mod.AscendQfaAttentionBackendImpl, method))
        ok_all &= check("REAL-CLASS-HIERARCHY", ok)
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        ok_all &= check("REAL-CLASS-HIERARCHY", False, f"{type(e).__name__}: {e}")

    print(f"[{'GREEN' if ok_all else 'RED'}] QFA startup import smoke "
          f"(real vllm @ {VLLM_SRC.name}, worktree {WORKTREE})")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
