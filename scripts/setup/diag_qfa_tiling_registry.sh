#!/usr/bin/env bash
# Diagnose the mass "do not registe tiling struct" failures in the junlin-qfa
# full build: distinguish (a) liboptiling.so carrying undefined symbols so the
# opc-side dlopen fails and every TILING_DATA_DEF op loses its registration,
# from (b) a stale TBE repository cache, from (c) an upstream baseline break.
#
# Read-only: inspects the failed build tree left in csrc/build and the newest
# pip_install log. Does NOT touch build dirs -- run this BEFORE any rebuild.
#
# Usage (inside the serving container):
#   bash scripts/setup/diag_qfa_tiling_registry.sh [worktree_dir]

set -uo pipefail

WORKTREE="${1:-/home/hajimi/qwen3.8/vllm-ascend/junlin-qfa}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

echo "[INFO] worktree=$WORKTREE"
echo "[INFO] HEAD=$(git -C "$WORKTREE" log --oneline -1 2>/dev/null || echo '?')"

fail=0

echo
echo "================ 1) tiling so: existence + unresolved symbols ================"
# The packaged tiling library is what opc dlopens to enumerate
# REGISTER_TILING_DATA_CLASS entries; any unresolved symbol kills the dlopen
# and takes every standard-mechanism op down with it.
found_so=0
while IFS= read -r so; do
    found_so=1
    real=$(readlink -f "$so")
    echo "[SO] $so"
    echo "     -> $(basename "$real") ($(stat -c '%s bytes, mtime %y' "$real" 2>/dev/null))"
    unresolved=$(ldd -r "$so" 2>&1 | grep -i "undefined" | c++filt | sort -u)
    if [[ -n "$unresolved" ]]; then
        count=$(printf '%s\n' "$unresolved" | wc -l)
        echo "[RED] $count unresolved symbol(s) in $(basename "$so"):"
        printf '%s\n' "$unresolved" | head -40
        fail=1
    else
        echo "[OK] no unresolved symbols"
    fi
done < <(find "$WORKTREE/csrc/build" "$WORKTREE/csrc/build_out" \
            \( -name "liboptiling.so" -o -name "libcust_opmaster_rt2.0.so" \
               -o -name "libcust_opapi.so" -o -name "libcust_opsproto_rt2.0.so" \) \
            2>/dev/null | sort -u)
if [[ $found_so -eq 0 ]]; then
    echo "[WARN] no tiling .so found under csrc/build{,_out} -- build tree already wiped?"
fi

echo
echo "================ 2) earliest real errors in the newest pip_install log ================"
LOG=$(ls -t "$LOG_DIR"/pip_install_*.log 2>/dev/null | head -1)
if [[ -n "${LOG:-}" && -f "$LOG" ]]; then
    echo "[LOG] $LOG"
    echo "-- first 15 error-ish lines (root cause is usually the FIRST, not the flood) --"
    grep -nE "error:|Error:|CMake Error|undefined symbol|undefined reference|cannot open shared|dlopen|ImportError|Traceback" "$LOG" | head -15
    echo "-- dlopen / tiling-so specific lines --"
    grep -nEi "dlopen|liboptiling|libcust_opmaster|do not registe" "$LOG" | head -10
else
    echo "[WARN] no pip_install log under $LOG_DIR"
fi

echo
echo "================ 3) TBE repository / kernel cache state ================"
# A half-failed earlier run can leave a stale op repository that later builds
# read back. List candidates; do NOT delete anything here.
for d in "$HOME/atc_data" /root/atc_data "$HOME/.ascend" /var/log/npu/slog; do
    if [[ -d "$d" ]]; then
        echo "[CACHE] $d ($(du -sh "$d" 2>/dev/null | cut -f1), newest: $(find "$d" -type f -newermt '-1 day' 2>/dev/null | wc -l) files touched <24h)"
    fi
done
compgen -G "$WORKTREE/csrc/build/binary/ascend950/gen/kernel_meta_*" >/dev/null 2>&1 \
    && echo "[CACHE] kernel_meta dirs present under csrc/build/binary/ascend950/gen"

echo
echo "================ 4) which ops actually registered in the built tiling so ================"
# If the so loads, its symbol table should carry one *TilingData ctor per
# standard-mechanism op; missing entries mean their tiling objs never linked in.
so=$(find "$WORKTREE/csrc/build" -name "liboptiling.so" 2>/dev/null | head -1)
if [[ -n "$so" ]]; then
    echo "-- TilingData-ish symbols per op (expect entries for hc_post/swiglu/rotary/kda/...) --"
    nm -DC "$(readlink -f "$so")" 2>/dev/null | grep -oE "[A-Za-z0-9]+TilingData" | sort | uniq -c | sort -rn | head -25
fi

echo
if [[ $fail -ne 0 ]]; then
    echo "[RED] tiling so carries unresolved symbols -- paste section 1 back for the fix"
    exit 1
fi
echo "[GREEN-ish] no unresolved symbols found; root cause likely in sections 2-4 -- paste them back"
