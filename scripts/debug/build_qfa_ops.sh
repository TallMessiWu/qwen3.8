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

# Third-party tarballs persist in csrc/third_party (outside build/), so wiping
# build/ does NOT re-download. If the cache is empty this build needs network
# access to gitcode.com once (abseil-cpp / protobuf) -- surface that up front.
if compgen -G "third_party/pkg/*" >/dev/null 2>&1 || [[ -d third_party/abseil-cpp/absl ]]; then
    echo "[INFO] third_party cache present: $(ls third_party/pkg 2>/dev/null | tr '\n' ' ')"
else
    echo "[WARN] no third_party cache under csrc/third_party -- first build will download abseil/protobuf from gitcode.com"
fi
# Mirror build_aclnn.sh: expose catlass headers via CPATH when the submodule
# is checked out (QFA itself does not need it, but keep the env identical).
if [[ -d third_party/catlass/include ]]; then
    export CPATH="$(cd third_party/catlass/include && pwd)${CPATH:+:${CPATH}}"
    echo "[INFO] catlass include exported to CPATH"
fi

rm -rf build output build_out

echo "[INFO] building (streams live below AND into $LOG)"
echo "[INFO] NOTE: after the host ninja finishes, the ascendc kernel phase (opc) is"
echo "[INFO] SILENT and can run 10-40+ minutes; the heartbeat below proves liveness."
(
    start=$(date +%s)
    while sleep 60; do
        objs=$(find build/binary -name '*.o' 2>/dev/null | wc -l)
        procs=$(pgrep -fc 'ccec|bisheng|opc|te_fusion' 2>/dev/null || echo 0)
        echo "[hb +$((($(date +%s) - start) / 60))min] alive: kernel objs=$objs, compiler procs=$procs"
    done
) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null || true' EXIT
set +e
bash build.sh --pkg --ops="quant_flash_attn;quant_flash_attn_metadata" --soc="$SOC" 2>&1 | tee "$LOG"
rc=$?
set -e
kill "$HB_PID" 2>/dev/null || true

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
    grep -nE "error:|Error:|CMake Error|undefined reference|undefined symbol|No such file|do not registe|ld\.lld|\[ERROR\]" "$LOG" | head -40
    tiling_so="build/custom/op_impl/ai_core/tbe/op_tiling/liboptiling.so"
    if [[ -f "$tiling_so" ]]; then
        echo "================ ALL unresolved symbols in liboptiling.so (ldd -r) ================"
        ldd -r "$tiling_so" 2>&1 | grep -i "undefined" | c++filt | sort -u | head -30
    fi
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
# Mirror build_aclnn.sh: clear everything under _cann_ops_custom except
# .gitkeep so stale vendor content never mixes with this install.
install_dir="$WORKTREE/vllm_ascend/_cann_ops_custom"
mkdir -p "$install_dir"
find "$install_dir" -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec rm -rf -- {} +
chmod +x "$run_pkg" || true
bash "$run_pkg" --install-path="$install_dir" >>"$LOG" 2>&1 || {
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
