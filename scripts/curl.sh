#!/usr/bin/env bash

set -euo pipefail

IMAGE_PROMPT='请客观描述这张图片的内容，包括场景、主要主体、主体特征、动作以及主体之间的位置关系。不要编造图片中不可见的信息。'

BASE_DIR="${BASE_DIR:-/home/hajimi/qwen3.8/pics}"
IMAGE_SERVER_PORT="${IMAGE_SERVER_PORT:-6666}"
VLLM_PORT="${VLLM_PORT:-6969}"
MODEL_NAME="${MODEL_NAME:-qwen3.8}"
CHAT_URL="http://127.0.0.1:${VLLM_PORT}/v1/chat/completions"
SERVER_PID=

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[System] 清理后台 HTTP 服务 (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    exit "$status"
}

trap cleanup EXIT INT TERM

for image_name in outdoor-courtyard.png indoor-kitchen.png; do
    if [[ ! -r "${BASE_DIR}/${image_name}" ]]; then
        echo "ERROR: missing or unreadable image ${BASE_DIR}/${image_name}" >&2
        exit 2
    fi
done

echo "[System] 正在启动本地 HTTP 静态文件服务 (端口: $IMAGE_SERVER_PORT)..."
python3 -m http.server "$IMAGE_SERVER_PORT" --directory "$BASE_DIR" >/dev/null 2>&1 &
SERVER_PID=$!

image_server_ready=0
for _ in {1..20}; do
    if curl -fsS "http://127.0.0.1:${IMAGE_SERVER_PORT}/outdoor-courtyard.png" -o /dev/null 2>/dev/null; then
        image_server_ready=1
        break
    fi
    sleep 0.1
done
if [[ "$image_server_ready" -ne 1 ]]; then
    echo "ERROR: local image server did not become ready on port $IMAGE_SERVER_PORT" >&2
    exit 2
fi
echo "[System] 本地 HTTP 服务已就绪！"

send_image_request() {
    local label="$1"
    local image_name="$2"
    local IMAGE_URL="http://127.0.0.1:${IMAGE_SERVER_PORT}/${image_name}"

    echo "[$label] 正在发送图片 ${image_name} ..."
    jq -n \
        --arg model "$MODEL_NAME" \
        --arg prompt "$IMAGE_PROMPT" \
        --arg uri "$IMAGE_URL" \
        '{
          model: $model,
          temperature: 0,
          top_p: 0.95,
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: $prompt },
                { type: "image_url", image_url: { url: $uri } }
              ]
            }
          ]
        }' | curl -sS "$CHAT_URL" \
        -H "Content-Type: application/json" \
        --data-binary @-
    printf '\n\n'
    sleep 2
}

send_image_request "Test 1" "outdoor-courtyard.png"
send_image_request "Test 2" "indoor-kitchen.png"

echo "[Test 3] 正在发送纯文本请求 1 (身份问答) ..."
jq -n \
    --arg model "$MODEL_NAME" \
    '{
      model: $model,
      temperature: 0,
      top_p: 0.95,
      min_p: 0,
      messages: [
        {
          role: "user",
          content: "你好啊？你叫什么名字？"
        }
      ]
    }' | curl -sS "$CHAT_URL" \
    -H "Content-Type: application/json" \
    --data-binary @-

printf '\n\n'
sleep 2

echo "[Test 4] 正在发送纯文本请求 2 (JoJo 知识问答) ..."
jq -n \
    --arg model "$MODEL_NAME" \
    '{
      model: $model,
      temperature: 0,
      max_tokens: 500,
      top_p: 0.95,
      min_p: 0,
      messages: [
        {
          role: "user",
          content: "解释一下JoJo的奇妙冒险里面败者食尘能力是什么。"
        }
      ]
    }' | curl -sS "$CHAT_URL" \
    -H "Content-Type: application/json" \
    --data-binary @-

printf '\n'
