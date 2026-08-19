#!/bin/bash

IMAGE_PROMPT='请客观描述这张图片的内容，包括场景、主要主体、主体特征、动作以及主体之间的位置关系。不要编造图片中不可见的信息。'

##########################################################################################

# 1. 定义本地图片路径
IMAGE_PATH="/home/hajimi/qwen3.8/pics/outdoor-courtyard.png"

# 2. 将本地图片转换为 Base64 编码（去除换行符）
# 注意冷知识：Linux 系统使用 base64 -w 0，而 macOS 系统必须使用 base64 -b 0 或 base64 -i
IMAGE_B64=$(base64 -w 0 "$IMAGE_PATH")

# 3. 构造 PNG Data URI
DATA_URI="data:image/png;base64,${IMAGE_B64}"

# 使用 jq 构造安全的 JSON Payload
PAYLOAD=$(jq -n \
  --arg prompt "$IMAGE_PROMPT" \
  --arg uri "$DATA_URI" \
  '{
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
  }')

# 发送请求
curl http://127.0.0.1:${VLLM_PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD"

echo -e "\n"

sleep 2

##########################################################################################

# 1. 定义本地图片路径
IMAGE_PATH="/home/hajimi/qwen3.8/pics/indoor-kitchen.png"

# 2. 将本地图片转换为 Base64 编码（去除换行符）
# 注意冷知识：Linux 系统使用 base64 -w 0，而 macOS 系统必须使用 base64 -b 0 或 base64 -i
IMAGE_B64=$(base64 -w 0 "$IMAGE_PATH")

# 3. 构造 PNG Data URI
DATA_URI="data:image/png;base64,${IMAGE_B64}"

# 使用 jq 构造安全的 JSON Payload
PAYLOAD=$(jq -n \
  --arg prompt "$IMAGE_PROMPT" \
  --arg uri "$DATA_URI" \
  '{
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
  }')

# 发送请求
curl http://127.0.0.1:${VLLM_PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD"

echo -e "\n"

sleep 2

##########################################################################################

curl http://127.0.0.1:${VLLM_PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 0,
    "top_p": 0.95,
    "min_p": 0,
    "messages": [
      {
        "role": "user",
        "content": "你好啊？你叫什么名字？"
      }
    ]
  }'

echo -e "\n"

sleep 2

##########################################################################################

curl http://127.0.0.1:${VLLM_PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 0,
    "max_tokens": 500,
    "top_p": 0.95,
    "min_p": 0,
    "messages": [
      {
        "role": "user",
        "content": "解释一下JoJo的奇妙冒险里面败者食尘能力是什么。"
      }
    ]
  }'
