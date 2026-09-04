#!/usr/bin/env bash
# 在【本机】（x86_64 + NVIDIA GPU，无 NPU、无 CANN）重建 Qwen3.8 的静态检查 /
# 数值模拟环境。这不是服务器脚本——服务器侧请用 scripts/setup/install-vllm-ascend.sh。
#
# 环境构成，刻意对齐真机容器（py311 + vllm 0.27.1 + torch 2.10.0）：
#   - Python 3.11
#   - vllm 0.27.1，从主仓 vllm submodule 的 v0.27.1 worktree 以 VLLM_TARGET_DEVICE=empty
#     安装：不编译任何 CUDA kernel，不需要 nvcc，装的是与真机一致的那份 Python 源码
#   - torch 2.10.0（PyPI 默认 wheel，自带 CUDA，可用本机 GPU 跑数值等价性验证）
#   - vllm-ascend 以 editable 方式指向 junlin-qfa worktree，--no-deps 跳过 torch-npu /
#     triton-ascend 这些本机装不了也用不上的依赖
#
# torch_npu 不安装：vllm-ascend 的 tests/ut/conftest.py 会在探测不到 npu-smi 时
# 自动注入 MagicMock 版 torch_npu / acl / triton.runtime，这正是 CI 的 CPU runner 口径。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLLM_REF="${VLLM_REF:-v0.27.1}"
VLLM_WORKTREE="${VLLM_WORKTREE:-$REPO_ROOT/.dev/vllm-0.27.1}"
ASCEND_WORKTREE="${ASCEND_WORKTREE:-$REPO_ROOT/vllm-ascend/junlin-qfa}"
SOC_VERSION="${SOC_VERSION:-ascend910b1}"

cd "$REPO_ROOT"

command -v uv >/dev/null || { echo "ERROR: 需要 uv，见 https://docs.astral.sh/uv/" >&2; exit 1; }
[[ -d "$ASCEND_WORKTREE" ]] || { echo "ERROR: 找不到 vllm-ascend worktree: $ASCEND_WORKTREE" >&2; exit 1; }

echo "==> [1/5] vllm $VLLM_REF 只读 worktree"
if [[ -d "$VLLM_WORKTREE" ]]; then
    echo "    已存在，跳过：$VLLM_WORKTREE"
else
    mkdir -p "$(dirname "$VLLM_WORKTREE")"
    # 从 submodule 仓库派生 worktree，vllm/ 本身的 checkout 一个字节都不动
    git -C "$REPO_ROOT/vllm" worktree add "$VLLM_WORKTREE" "$VLLM_REF"
fi

echo "==> [2/5] 创建 venv（Python 3.11）"
uv venv --python 3.11

echo "==> [3/5] 同步 PyPI 依赖（torch + 测试/lint 工具）"
# --inexact：不要移除后面用 uv pip install 装进来的 vllm / vllm-ascend
uv sync --inexact --group test --group lint

echo "==> [4/5] 安装 vllm（VLLM_TARGET_DEVICE=empty，不编译 kernel）"
# 不加 --no-deps：empty target 下 setup.py 只读 requirements/common.txt（不含 torch pin），
# 这些是 import vllm 真正要用到的运行期依赖，缺了 patch 检查跑不起来
VLLM_TARGET_DEVICE=empty uv pip install "$VLLM_WORKTREE"

echo "==> [5/5] 安装 vllm-ascend（editable，无设备）"
# --no-build-isolation 要求构建依赖已经在环境里；走隔离构建的话 uv 会照着
# pyproject 的 build-system.requires 去装 torch-npu，本机既装不出能用的东西又白下 1GB
uv pip install --quiet "setuptools>=64" "setuptools-scm>=8" wheel
# --no-deps 跳过 torch-npu / triton-ascend：本机没有 CANN，装了也 import 不起来
SOC_VERSION="$SOC_VERSION" COMPILE_CUSTOM_KERNELS=0 \
    uv pip install --no-deps --no-build-isolation -e "$ASCEND_WORKTREE"

echo
echo "==> 自检"
"$REPO_ROOT/.venv/bin/python" - <<'PY'
import importlib.metadata as md
import importlib.util
import torch

print(f"python      : {__import__('sys').version.split()[0]}")
print(f"torch       : {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f"gpu         : {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}")
print(f"vllm        : {md.version('vllm')}  ({importlib.util.find_spec('vllm').origin})")
print(f"vllm-ascend : {md.version('vllm-ascend')}  ({importlib.util.find_spec('vllm_ascend').origin})")
PY

cat <<'TIP'

用法：
  python scripts/local/check_patch_targets.py   # patch 目标还在不在、参数列表有没有漂
  bash   scripts/local/run_cpu_ut.sh            # CPU 单测 + 跟已知基线比对

能拦什么、拦不住什么、已知的基线失败：scripts/local/README.md
TIP
