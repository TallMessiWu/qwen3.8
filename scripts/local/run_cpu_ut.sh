#!/usr/bin/env bash
# 在本机跑 vllm-ascend 的 CPU 单测，并跟已知基线比对。
#
# 关注点是【新增】失败：本机 mock 出来的环境跟真机总有出入，有几个用例本来就红，
# 单看"有没有失败"没意义，看"比基线多红了哪些"才有意义。
#
#   bash scripts/local/run_cpu_ut.sh                   # 跑全部 CPU 用例并比对基线
#   bash scripts/local/run_cpu_ut.sh tests/ut/ops      # 只跑一部分（不比对基线）
#   UPDATE_BASELINE=1 bash scripts/local/run_cpu_ut.sh # 确认过之后刷新基线

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKTREE="${WORKTREE:-$REPO_ROOT/vllm-ascend/junlin-qfa}"
BASELINE="${BASELINE:-$REPO_ROOT/scripts/local/ut_baseline.txt}"
PYTHON="$REPO_ROOT/.venv/bin/python"

[[ -x "$PYTHON" ]] || { echo "ERROR: 先跑 scripts/local/setup-devenv.sh" >&2; exit 1; }
cd "$WORKTREE"

# tests/ut/<module>/{a2,a2_2,a3_2,a3_4,310p} 按约定是 NPU 专属，本机不跑
mapfile -t npu_dirs < <(find tests/ut -type d \
    \( -name a2 -o -name a2_2 -o -name a3_2 -o -name a3_4 -o -name 310p -o -name _310p \))
ignores=()
for d in "${npu_dirs[@]}"; do ignores+=("--ignore=$d"); done

targets=("$@")
partial=1
if [[ ${#targets[@]} -eq 0 ]]; then
    targets=(tests/ut)
    partial=0
fi

log=$(mktemp)
trap 'rm -f "$log"' EXIT
"$PYTHON" -m pytest -q -p no:cacheprovider "${ignores[@]}" "${targets[@]}" 2>&1 | tee "$log"

if [[ $partial -eq 1 ]]; then
    echo
    echo "只跑了部分用例，跳过基线比对。"
    exit 0
fi

current=$(grep -E "^FAILED |^ERROR " "$log" | awk '{print $2}' | sort -u)

if [[ "${UPDATE_BASELINE:-0}" == "1" ]]; then
    printf '%s\n' "$current" > "$BASELINE"
    echo "基线已刷新：$BASELINE（$(printf '%s\n' "$current" | grep -c . ) 条）"
    exit 0
fi

if [[ ! -f "$BASELINE" ]]; then
    echo "没有基线文件，先跑一次 UPDATE_BASELINE=1 建立它。"
    exit 0
fi

known=$(grep -vE "^\s*#|^\s*$" "$BASELINE" | sort -u)
new=$(comm -13 <(printf '%s\n' "$known") <(printf '%s\n' "$current") | grep -c . )
fixed=$(comm -23 <(printf '%s\n' "$known") <(printf '%s\n' "$current") | grep -c . )

echo
if [[ "$new" -gt 0 ]]; then
    echo "RED   比基线新增 $new 条失败："
    comm -13 <(printf '%s\n' "$known") <(printf '%s\n' "$current") | sed 's/^/        /'
fi
if [[ "$fixed" -gt 0 ]]; then
    echo "GREEN 基线里有 $fixed 条现在过了（确认无误后 UPDATE_BASELINE=1 刷新）："
    comm -23 <(printf '%s\n' "$known") <(printf '%s\n' "$current") | sed 's/^/        /'
fi
[[ "$new" -eq 0 && "$fixed" -eq 0 ]] && echo "GREEN 与基线一致，没有新增失败。"

exit $(( new > 0 ? 1 : 0 ))
