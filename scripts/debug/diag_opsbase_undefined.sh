#!/usr/bin/env bash
# Diagnose the liboptiling.so "undefined symbol: Ops::Base::*" class of failures.
#
# Root-cause hypothesis to confirm/refute: the custom tiling library
# (build/custom/op_impl/ai_core/tbe/op_tiling/liboptiling.so -> libcust_opmaster_rt2.0.so)
# is linked WITHOUT CANN's libops_base.so (FindOPBASE.cmake failed to locate it),
# leaving Ops::Base::ToString/CeilDiv/... unresolved at dlopen time.
#
# Prints a single report. No NPU / weights / credentials involved.
#
# Usage (inside the serving container):
#   bash scripts/debug/diag_opsbase_undefined.sh [worktree_dir] [pip_install_log]
#     worktree_dir   default /home/hajimi/qwen3.8/vllm-ascend/junlin-qfa
#     pip_install_log optional; grep it for the OPBASE find result
set -uo pipefail

WORKTREE="${1:-/home/hajimi/qwen3.8/vllm-ascend/junlin-qfa}"
LOG="${2:-}"
ASCEND_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend}"

echo "================ 1. locate libops_base.so under $ASCEND_HOME ================"
found=$(find "$ASCEND_HOME" -name 'libops_base*.so*' 2>/dev/null | sort)
if [[ -z "$found" ]]; then
    echo "[RED] libops_base.so NOT FOUND anywhere under $ASCEND_HOME"
    echo "      -> the ops_base CANN component is missing; install it (e.g. Ascend-cann-opsbase-*)"
else
    echo "$found"
    echo "[GREEN] libops_base.so present; check the exact path vs FindOPBASE search paths below"
fi

echo
echo "================ 2. FindOPBASE search paths (what the build looked for) ================"
# SYSTEM_PREFIX is <arch>-linux; the build searched ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
arch=$(uname -m)
echo "SYSTEM_PREFIX=${arch}-linux"
echo "expected lib path: \${ASCEND_DIR}/${arch}-linux/lib64/libops_base.so"
for asc in $(find "$ASCEND_HOME" -maxdepth 2 -name 'ascend-toolkit' -o -maxdepth 1 -name 'cann-*' 2>/dev/null); do
    echo "  check: $asc/${arch}-linux/lib64/libops_base.so -> $([[ -e "$asc/${arch}-linux/lib64/libops_base.so" ]] && echo FOUND || echo missing)"
done

echo
echo "================ 3. OPBASE result in build log ================"
if [[ -n "$LOG" && -f "$LOG" ]]; then
    grep -nE "OPBASE|ops_base|Cannot find library ops_base|Found OPABSE" "$LOG" | head -20 || echo "(no OPBASE lines in log)"
else
    echo "(no log given; pass the pip_install log path as \$2)"
fi

echo
echo "================ 4. ALL unresolved symbols in the tiling .so (ldd -r) ================"
so="$WORKTREE/csrc/build/custom/op_impl/ai_core/tbe/op_tiling/liboptiling.so"
if [[ ! -e "$so" ]]; then
    echo "[SKIP] $so not built yet"
else
    real=$(readlink -f "$so")
    echo "tiling .so -> $real"
    ldd -r "$real" 2>&1 | grep -iE "undefined|not found" | c++filt | sort -u | head -60
    echo "---- summary ----"
    n=$(ldd -r "$real" 2>&1 | grep -ciE "undefined")
    echo "total undefined-symbol lines: $n"
    if [[ "$n" -eq 0 ]]; then
        echo "[GREEN] no unresolved symbols"
    else
        echo "[RED] $n unresolved symbols present"
    fi
fi
