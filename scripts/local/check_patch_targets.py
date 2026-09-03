#!/usr/bin/env python3
"""检查 vllm-ascend 的 monkey-patch 目标在当前 vllm 里是否还对得上。

本机（无 NPU）能拦住的最有价值的一类错误：vllm 升级后 patch 指向的函数/方法
被改名、挪走或改了签名。前两种在 import 期就会炸，第三种最阴——import 不报错、
MagicMock 也放过，一到真机就崩或静默走错分支。

做法：
  1. AST 扫描 vllm_ascend/patch/ 下所有模块级的 ``Target.attr = replacement`` 赋值，
     结合该文件的 import 语句还原 Target 的完整路径；
  2. 在 patch 生效【前】取 vllm 侧原属性的签名；
  3. 触发 adapt_patch()，取替换实现的签名；
  4. 比对参数列表，报告缺失目标与签名漂移。

判据：
  RED   patch 模块加载了，但目标前后都不存在 —— 这处 patch 打空了
  AMBER 参数列表对不上 —— 多数是有意适配，但每条都该能说出为什么
  NEW   patch 前没有这个名字 —— 有意新增，或目标改名后 patch 静默打空，需逐条确认
  SKIP  承载 patch 的模块本机没加载（HAS_TRITON 之类的守卫）—— 只有真机能判
  GREEN 目标存在且参数一致

用法（在仓库根，激活 .venv 后）：
    python scripts/local/check_patch_targets.py
    python scripts/local/check_patch_targets.py --worktree vllm-ascend/main
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PatchSite:
    """一处 ``Target.attr = replacement`` 赋值。"""

    file: Path
    lineno: int
    target_root: str  # 源码里写的根名字，如 ``utils``
    target_path: str  # 还原后的完整路径，如 ``vllm.utils``
    attr: str  # 被替换的属性名
    replacement: str  # 替换实现在 patch 模块里的名字

    @property
    def dotted(self) -> str:
        return f"{self.target_path}.{self.attr}"


def _resolve_imports(tree: ast.Module) -> dict[str, str]:
    """把 patch 文件里的 import 收成 {本地名: 完整路径}。"""
    mapping: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mapping[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                mapping[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return mapping


def collect_patch_sites(patch_dir: Path) -> list[PatchSite]:
    sites: list[PatchSite] = []
    for path in sorted(patch_dir.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # 语法错误本身就是要拦的错
            print(f"RED   {path}: 无法解析 —— {exc}")
            continue
        imports = _resolve_imports(tree)
        # 只看模块级赋值：函数体内的赋值多是条件 patch，静态还原不可靠
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target, value = node.targets[0], node.value
            if not isinstance(target, ast.Attribute) or not isinstance(target.value, ast.Name):
                continue
            if not isinstance(value, (ast.Name, ast.Attribute)):
                continue
            root = target.value.id
            resolved = imports.get(root)
            # 只关心打到 vllm 上的；vllm_ascend 打自己不算 patch 目标
            if not resolved or not (resolved == "vllm" or resolved.startswith("vllm.")):
                continue
            sites.append(
                PatchSite(
                    file=path,
                    lineno=node.lineno,
                    target_root=root,
                    target_path=resolved,
                    attr=target.attr,
                    replacement=ast.unparse(value),
                )
            )
    return sites


def install_npu_stubs(worktree: Path) -> None:
    """复用 vllm-ascend 自己的 torch_npu mock，而不是另抄一份。

    ``tests/ut/conftest.py`` 的结构是「先注入 mock，再 import pytest，最后 adapt_patch()」。
    我们只执行 ``import pytest`` 之前的那段——拿到跟 CI CPU runner 完全一致的 mock，
    又不会提前触发 patch（否则就取不到 patch 前的原始签名了）。
    conftest 改结构时这里会显式报错，不会悄悄失效。
    """
    conftest = worktree / "tests" / "ut" / "conftest.py"
    if not conftest.is_file():
        raise SystemExit(f"RED   找不到 conftest：{conftest}")

    tree = ast.parse(conftest.read_text(encoding="utf-8"))
    cut = next(
        (
            i
            for i, node in enumerate(tree.body)
            if isinstance(node, ast.Import) and any(a.name == "pytest" for a in node.names)
        ),
        None,
    )
    if cut is None:
        raise SystemExit(f"RED   {conftest} 结构变了：找不到 `import pytest` 这道分界线")

    prelude = ast.Module(body=tree.body[:cut], type_ignores=[])
    # exec 是这里的重点：目的就是执行 conftest 自己那段 mock，而不是另抄一份跟着漂
    exec(  # noqa: S102
        compile(prelude, str(conftest), "exec"),
        {"__name__": "_ut_conftest_prelude", "__file__": str(conftest)},
    )


def _patch_module_name(file: Path, worktree: Path) -> str:
    """``…/vllm_ascend/patch/worker/patch_triton.py`` -> ``vllm_ascend.patch.worker.patch_triton``"""
    return file.relative_to(worktree).with_suffix("").as_posix().replace("/", ".")


def _lookup(dotted: str):
    """按 ``a.b.c`` 逐段解析，模块优先、属性兜底。"""
    parts = dotted.split(".")
    obj = None
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for part in parts[i:]:
            obj = getattr(obj, part)
        return obj
    raise ImportError(dotted)


def _params(obj) -> list[tuple[str, int]] | None:
    """取参数名 + 传参方式。

    刻意不比较类型注解和返回类型：``from __future__ import annotations`` 会让一边
    是字符串一边是对象，patch 实现也常常省掉返回注解——这些都不影响调用能否成立。
    真正会在真机上崩的是参数名和位置/关键字属性对不上。
    """
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    return [(name, p.kind.value) for name, p in sig.parameters.items()]


def _describe_drift(old: list[tuple[str, int]], new: list[tuple[str, int]]) -> str:
    old_names = [n for n, _ in old]
    new_names = [n for n, _ in new]
    added = [n for n in new_names if n not in old_names]
    removed = [n for n in old_names if n not in new_names]
    bits = []
    if removed:
        bits.append("少了 " + ", ".join(removed))
    if added:
        bits.append("多了 " + ", ".join(added))
    if not bits:  # 参数名一致，那就是 kind 或顺序变了
        kinds = {"0": "仅位置", "1": "位置或关键字", "2": "*args", "3": "仅关键字", "4": "**kwargs"}
        changed = [
            f"{n}({kinds.get(str(ok), ok)}->{kinds.get(str(nk), nk)})"
            for (n, ok), (_, nk) in zip(old, new, strict=False)
            if ok != nk
        ]
        bits.append("传参方式变化 " + ", ".join(changed) if changed else "参数顺序变化")
    return "；".join(bits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--worktree",
        default="vllm-ascend/junlin-qfa",
        help="要检查的 vllm-ascend worktree（相对仓库根），默认 junlin-qfa",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="AMBER 同时打印完整参数列表")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    patch_dir = repo_root / args.worktree / "vllm_ascend" / "patch"
    if not patch_dir.is_dir():
        print(f"RED   找不到 patch 目录：{patch_dir}")
        return 2

    # 必须在 import vllm 之前：否则平台探测会选中 CUDA，而本机 vllm 是
    # VLLM_TARGET_DEVICE=empty 构建的，没有 _C_stable_libtorch，import 直接炸
    sys.path.insert(0, str(repo_root / args.worktree))
    install_npu_stubs(repo_root / args.worktree)

    import vllm

    print(f"vllm {vllm.__version__} @ {Path(vllm.__file__).parent}")
    print(f"patch 目录 {patch_dir.relative_to(repo_root)}\n")

    sites = collect_patch_sites(patch_dir)
    print(f"扫到 {len(sites)} 处打向 vllm 的模块级 patch 赋值\n")

    # 第一步：patch 生效前，记录 vllm 侧原始签名
    missing: list[tuple[PatchSite, str]] = []
    before: dict[str, list[tuple[str, int]] | None] = {}
    for site in sites:
        try:
            owner = _lookup(site.target_path)
        except (ImportError, AttributeError) as exc:
            missing.append((site, f"目标模块/类不存在：{exc}"))
            continue
        if not hasattr(owner, site.attr):
            missing.append((site, f"{site.target_path} 上没有属性 {site.attr}"))
            continue
        before[site.dotted] = _params(getattr(owner, site.attr))

    # 第二步：触发 patch，取替换实现的签名
    from vllm_ascend.utils import adapt_patch

    adapt_patch(True)
    adapt_patch()

    drifted: list[tuple[PatchSite, list, list]] = []
    for site in sites:
        if site.dotted not in before:
            continue
        try:
            after = _params(getattr(_lookup(site.target_path), site.attr))
        except (ImportError, AttributeError):
            continue
        old = before[site.dotted]
        if old is not None and after is not None and old != after:
            drifted.append((site, old, after))

    # 复查 missing，分三种情况：
    #   SKIP 承载这处 patch 的模块压根没被加载——patch/worker/__init__.py 里有
    #        `if HAS_TRITON:` 这类守卫，而 conftest 把 triton.runtime 换成了
    #        MagicMock，本机 HAS_TRITON 恒为 False。这类只有真机能判。
    #   NEW  模块加载了，属性 patch 后才出现。注意这【不是】安全信号：monkeypatch
    #        赋值一定会把属性创建出来，所以「目标改名了、patch 往废名字上赋值、
    #        静默失效」看起来跟「有意往 vllm 挂新属性」一模一样。必须逐条确认。
    #   RED  模块加载了，属性仍不存在：这处 patch 真的打空了。
    skipped: list[tuple[PatchSite, str]] = []
    added: list[tuple[PatchSite, str]] = []
    really_missing: list[tuple[PatchSite, str]] = []
    for site, why in missing:
        if _patch_module_name(site.file, repo_root / args.worktree) not in sys.modules:
            skipped.append((site, why))
            continue
        try:
            exists = hasattr(_lookup(site.target_path), site.attr)
        except (ImportError, AttributeError):
            exists = False
        (added if exists else really_missing).append((site, why))

    # RED 先出，它才是必须立刻处理的
    for site, why in really_missing:
        rel = site.file.relative_to(repo_root)
        print(f"RED   {rel}:{site.lineno}  {site.dotted}")
        print(f"      {why}  —— patch 之后依然不存在，这处 patch 打空了")
    if really_missing:
        print()
    for site, _ in added:
        rel = site.file.relative_to(repo_root)
        print(f"NEW   {rel}:{site.lineno}  {site.dotted}")
        print("      patch 前 vllm 上没有这个名字。可能是有意新增属性，也可能是目标已改名、")
        print("      patch 正往一个废名字上赋值而静默失效——必须确认是哪一种")
    if added:
        print()
    for site, _ in skipped:
        rel = site.file.relative_to(repo_root)
        mod = _patch_module_name(site.file, repo_root / args.worktree)
        print(f"SKIP  {rel}:{site.lineno}  {site.dotted}")
        print(f"      {mod} 未被加载（本机 HAS_TRITON=False 之类的守卫挡掉了），只有真机能判")
    if skipped and drifted:
        print()
    for site, old, new in drifted:
        rel = site.file.relative_to(repo_root)
        print(f"AMBER {rel}:{site.lineno}  {site.dotted}")
        print(f"      {_describe_drift(old, new)}")
        if args.verbose:
            print(f"      vllm 原  {[n for n, _ in old]}")
            print(f"      patch 后 {[n for n, _ in new]}")

    ok = len(sites) - len(missing) - len(drifted)
    print(f"\nGREEN {ok}  AMBER {len(drifted)}  NEW {len(added)}  SKIP {len(skipped)}  RED {len(really_missing)}")
    if drifted:
        print("AMBER 是参数列表对不上：多数是有意适配，但每一条都该能说出为什么。")
    if added:
        print("NEW 不等于安全：赋值一定会成功，所以它也可能是目标改名后 patch 静默打空。")
    return 1 if really_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
