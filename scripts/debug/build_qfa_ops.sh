#!/usr/bin/env bash
# Fast-iteration build for the two QFA custom ops with readable error extraction.
#
# Builds ONLY quant_flash_attn + quant_flash_attn_metadata (their attention/common
# dependency is pulled in automatically), tees the full log to a file, and on
# failure prints every ninja "FAILED:" block plus surrounding compiler errors so
# the root cause is not buried under parallel-job warnings. On success installs
# the .run package into vllm_ascend/_cann_ops_custom and nm-checks the aclnn
# symbols. Ends with [GREEN]/[RED].
#
# Usage (inside the serving container):
#   bash scripts/debug/build_qfa_ops.sh [worktree_dir]
# Default worktree: /home/hajimi/qwen3.8/vllm-ascend/feat-qfa-mxfp8-attn

set -uo pipefail

WORKTREE="${1:-/home/hajimi/qwen3.8/vllm-ascend/feat-qfa-mxfp8-attn}"
SOC="${QFA_BUILD_SOC:-ascend950}"
LOG="/tmp/qfa_build_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -d "$WORKTREE/csrc" ]]; then
    echo "[RED] missing $WORKTREE/csrc" >&2
    exit 2
fi

cd "$WORKTREE/csrc"
echo "[INFO] worktree=$WORKTREE soc=$SOC log=$LOG"
echo "[INFO] HEAD=$(git -C "$WORKTREE" log --oneline -1)"
rm -rf build output build_out

set +e
bash build.sh --pkg --ops="quant_flash_attn;quant_flash_attn_metadata" --soc="$SOC" >"$LOG" 2>&1
rc=$?
set -e

if [[ $rc -ne 0 ]]; then
    echo "[RED] build failed (exit=$rc). Extracting failures from $LOG:"
    echo "================ ninja FAILED blocks ================"
    # Each ninja failure: "FAILED: <target>" + command + compiler stderr.
    # Print each block until the next progress line / next FAILED / ninja summary.
    awk '
        /^FAILED: / { inblock=1; nblock++; print "\n---- FAILED block " nblock " ----"; print; next }
        inblock && (/^\[[0-9]+\/[0-9]+\]/ || /^ninja: /) { inblock=0 }
        inblock { print }
    ' "$LOG" | head -300
    echo "================ error lines (with context) ================"
    grep -nE "error:|Error:|CMake Error|undefined reference|No such file" "$LOG" | head -40
    echo "================ last 30 lines ================"
    tail -30 "$LOG"
    echo "[RED] full log: $LOG"
    exit 1
fi

echo "[OK] build succeeded, installing package"
run_pkg=$(ls build/cann-ops-transformer*.run 2>/dev/null | head -1)
if [[ -z "$run_pkg" ]]; then
    echo "[RED] no .run package under csrc/build (see $LOG)" >&2
    exit 1
fi
bash "$run_pkg" --install-path="$WORKTREE/vllm_ascend/_cann_ops_custom" >>"$LOG" 2>&1 || {
    echo "[RED] .run install failed, tail of log:"; tail -20 "$LOG"; exit 1;
}

lib="$WORKTREE/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so"
if [[ ! -f "$lib" ]]; then
    echo "[RED] $lib not found after install" >&2
    exit 1
fi
echo "[INFO] aclnn symbols in $(basename "$lib"):"
syms=$(nm -D "$lib" | grep -ci "QuantFlashAttn" || true)
nm -D "$lib" | grep -i "QuantFlashAttn" | head -8
if [[ "$syms" -lt 4 ]]; then
    echo "[RED] expected >=4 QuantFlashAttn symbols (main+metadata, GetWorkspaceSize+exec), got $syms"
    exit 1
fi
echo "[GREEN] QFA ops built + installed, $syms symbols exported (full log: $LOG)"
