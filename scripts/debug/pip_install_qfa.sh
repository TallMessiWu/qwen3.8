#!/usr/bin/env bash
# Full editable install of vllm-ascend (all custom ops + vllm_ascend_C torch
# bindings) with the log captured and the real errors extracted at the end.
#
# Mid-stream [ERROR] lines are often cascade noise from TBE; what matters is
# whether any "Op[...] compile failed" / ninja FAILED / pip build error remains
# at the end -- this script prints exactly those sections for handoff.
#
# Usage (inside the serving container):
#   bash scripts/debug/pip_install_qfa.sh [worktree_dir]
# Env: SOC_VERSION (default ascend950pr_9589)

set -uo pipefail

WORKTREE="${1:-/home/hajimi/qwen3.8/vllm-ascend/feat-qfa-mxfp8-attn}"
export SOC_VERSION="${SOC_VERSION:-ascend950pr_9589}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/pip_install_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -f "$WORKTREE/setup.py" ]]; then
    echo "[RED] missing $WORKTREE/setup.py" >&2
    exit 2
fi

cd "$WORKTREE"
echo "[INFO] worktree=$WORKTREE SOC_VERSION=$SOC_VERSION log=$LOG"
echo "[INFO] HEAD=$(git log --oneline -1)"
# The quick 2-op script leaves csrc/build/custom holding a QFA-only op store;
# reusing it makes every other op fall through to CANN's built-in catalog and
# fail parameter checks (e.g. SwigluGroupQuant 3-vs-4 inputs). Default to a
# clean build; third-party tarballs live outside build/ so nothing re-downloads.
if [[ "${KEEP_BUILD:-0}" != "1" ]]; then
    echo "[INFO] wiping csrc/{build,build_out,output} for an un-mixed full build (KEEP_BUILD=1 to reuse)"
    rm -rf csrc/build csrc/build_out csrc/output
fi
echo "[INFO] full pip output goes to the log; only progress/error lines show live."
echo "[INFO] Ignore transient [ERROR] noise -- read the extracted sections at the end."

set +e
pip install -v --no-build-isolation -e . \
    -i https://mirrors.aliyun.com/pypi/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --trusted-host mirrors.aliyun.com \
    --trusted-host download.pytorch.org \
    --trusted-host download-r2.pytorch.org 2>&1 \
    | tee "$LOG" \
    | grep --line-buffered -E "^\s*\[[0-9]+/[0-9]+\]|Generating |Opc tool|Building editable|Successfully|error|Error|FAILED|undefined|compile failed|\[ERROR\]|fatal"
rc=${PIPESTATUS[0]}
set -e

echo
if [[ $rc -ne 0 ]]; then
    echo "[RED] pip install failed (exit=$rc). Send the sections below:"
    echo "================ op kernel compile failures ================"
    grep -nE "Op\[[A-Za-z0-9_]+\].*compile failed|do not registe tiling struct" "$LOG" | sort -u | head -20
    echo "================ first compiler errors per file ================"
    grep -nE "error: |fatal error:" "$LOG" | awk -F: '!seen[$4]++' | head -30
    echo "================ undefined symbols ================"
    grep -nE "undefined symbol|undefined reference" "$LOG" | sort -u | head -15
    echo "================ last 25 lines ================"
    tail -25 "$LOG"
    echo "[RED] full log: $LOG"
    exit 1
fi

echo "[OK] pip install finished, verifying torch binding registration"
python - <<'EOF'
import torch  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

assert hasattr(torch.ops._C_ascend, "npu_quant_flash_attn"), "npu_quant_flash_attn missing"
assert hasattr(torch.ops._C_ascend, "npu_quant_flash_attn_metadata"), "metadata op missing"
print("[GREEN] torch.ops._C_ascend.npu_quant_flash_attn{,_metadata} registered")
EOF
echo "[GREEN] full install done (log: $LOG); next: python scripts/debug/test_quant_flash_attn_npu.py"
