#!/usr/bin/env bash
# GRAPH=1 / GRAPH=0 单变量 A/B：起服务 → 等就绪 → 发同一个长 prompt → 停服务 → 对拍插桩日志。
#
# 为什么要脚本而不是手敲：EP、MTP、QFA 一直开着，唯一在变的是 GRAPH，所以两次 run
# 的其他一切都必须逐字相同——prompt 尤其。prompt 在这里只生成一次、两次复用，
# 手工粘贴两遍容易差一个字符，而差一个字符 token 数就变了、结论就废了。
#
#   bash scripts/debug/ab_graph.sh              # 跑完整 A/B
#   REPS=110 bash scripts/debug/ab_graph.sh     # 换 prompt 长度
#   ONLY=1 bash scripts/debug/ab_graph.sh       # 只跑 GRAPH=1（机时不够时分两次做）
#
# 判据：GRAPH=1 应复现 completion=1（空输出），GRAPH=0 应为 completion=32。
# 若两边都正常，说明这次没复现，对拍无意义——别拿没复现的日志下结论。
#
# 日志和中间产物都落在本脚本所在目录的相对路径下（服务器根分区紧张，不写 /tmp）。

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/ab_out}"
mkdir -p "$OUT_DIR"

PORT="${VLLM_PORT:-6969}"
MODEL="${MODEL_NAME:-qwen3.8}"
REPS="${REPS:-68}"                  # 68 遍 ≈ 423 token，真机实测复现空输出的长度
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
ONLY="${ONLY:-}"

PROMPT_FILE="$OUT_DIR/prompt.txt"
python3 -c "
import sys
sys.stdout.write('人工智能的发展历程可以分为若干阶段。' * $REPS + ' 请用一句话总结。')
" > "$PROMPT_FILE"
echo "[ab] prompt 已生成：$(wc -c < "$PROMPT_FILE") 字节，两次 run 复用同一份"

SERVER_PID=""

stop_server() {
    [[ -z "$SERVER_PID" ]] && return 0
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[ab] 停服务 (PID $SERVER_PID)..."
        kill -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 60); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 2
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[ab] 进程没退干净，强杀" >&2
            kill -9 -- "-$SERVER_PID" 2>/dev/null || kill -9 "$SERVER_PID" 2>/dev/null || true
            echo "[ab] 若下一轮仍撞端口或显存不足，手动清一次：bash $SCRIPTS_DIR/npu-cleaner.sh all" >&2
        fi
    fi
    SERVER_PID=""
    # 等端口真正释放，否则下一轮起服务会撞端口
    for _ in $(seq 1 30); do
        curl -sS -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || break
        sleep 2
    done
    sleep 5
}

trap 'echo "[ab] 中断，清理中"; stop_server; exit 130' INT TERM
trap 'stop_server' EXIT

run_one() {
    local graph="$1"
    local log="$OUT_DIR/g${graph}.log"

    echo
    echo "=============== GRAPH=$graph ==============="
    # setsid 让服务自成进程组，停的时候能连同 8 个 worker 一起收掉
    setsid env MTP=3 QFA=1 GRAPH="$graph" bash "$SCRIPTS_DIR/397B.sh" > "$log" 2>&1 &
    SERVER_PID=$!
    echo "[ab] 服务已起 (PGID $SERVER_PID)，日志 $log"

    local waited=0
    until curl -sS -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[ab] RED：服务进程已退出，没起来。日志尾部：" >&2
            tail -30 "$log" >&2
            return 1
        fi
        if (( waited >= READY_TIMEOUT )); then
            echo "[ab] RED：等了 ${READY_TIMEOUT}s 仍未就绪，放弃。日志尾部：" >&2
            tail -30 "$log" >&2
            return 1
        fi
        sleep 10
        waited=$((waited + 10))
    done
    echo "[ab] 就绪（等了 ${waited}s），发 prompt"

    local resp
    # 用 --arg 读进来而不是 --rawfile：后者要 jq 1.6+，而 prompt 只有 ~1.5KB，
    # 走 shell 变量完全在 ARG_MAX 之内。prompt 末尾没有换行，$() 的截断无影响。
    resp=$(jq -n --arg m "$MODEL" --arg p "$(cat "$PROMPT_FILE")" \
        '{model:$m,temperature:0,max_tokens:32,messages:[{role:"user",content:$p}]}' \
        | curl -sS -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
               -H 'Content-Type: application/json' --data-binary @-)
    echo "$resp" > "$OUT_DIR/g${graph}.resp.json"
    echo "[ab] GRAPH=$graph 结果： $(echo "$resp" | jq -c '{prompt:.usage.prompt_tokens, completion:.usage.completion_tokens, finish:.choices[0].finish_reason}')"

    stop_server
    return 0
}

rc=0
if [[ "$ONLY" == "0" ]]; then
    run_one 0 || rc=1
elif [[ "$ONLY" == "1" ]]; then
    run_one 1 || rc=1
else
    run_one 1 || rc=1
    run_one 0 || rc=1
fi

echo
echo "=============== 对拍 ==============="
for g in 1 0; do
    log="$OUT_DIR/g${g}.log"
    [[ -f "$log" ]] || continue
    grep -o '\[moe-\(fp\|ag\|sel\|mc2\)\].*' "$log" > "$OUT_DIR/g${g}.tags"
    echo "g${g}: 日志 $(wc -l < "$log") 行，插桩 $(wc -l < "$OUT_DIR/g${g}.tags") 行" \
         "(fp=$(grep -c '\[moe-fp\]' "$log") ag=$(grep -c '\[moe-ag\]' "$log")" \
         "sel=$(grep -c '\[moe-sel\]' "$log") mc2=$(grep -c '\[moe-mc2\]' "$log"))"
done

if [[ -f "$OUT_DIR/g1.tags" && -f "$OUT_DIR/g0.tags" ]]; then
    echo
    if diff -q "$OUT_DIR/g1.tags" "$OUT_DIR/g0.tags" >/dev/null; then
        echo "两边插桩输出完全一致 —— MoE 这一段在图开关下没有任何差别，"
        echo "问题不在 MoE，得去查 aclgraph 改变的其他东西。"
    else
        echo "差异（前 40 行，左 g1 右 g0）："
        diff "$OUT_DIR/g1.tags" "$OUT_DIR/g0.tags" | head -40
    fi
fi

echo
echo "[ab] 产物都在 $OUT_DIR/"
exit $rc
