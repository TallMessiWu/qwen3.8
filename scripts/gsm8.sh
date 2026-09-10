#!/usr/bin/env bash
set -euo pipefail

# GSM8K 精度评测，跑在已经起好的 vLLM 服务上。
# 服务地址、端口和模型名写在 ais_bench 自带的 vllm_api_general_chat.py 里，
# 与本仓的服务入口对不上时改那个文件，不要改这里。
# --dump-eval-details 会把逐题的请求/回答落到 outputs/ 下，便于事后翻错题。
# 追加的参数原样透传给 ais_bench。

exec ais_bench \
    --models vllm_api_general_chat.py \
    --datasets gsm8k_gen_0_shot_cot_chat_prompt \
    --dump-eval-details \
    "$@"
