#!/usr/bin/env bash
set -euo pipefail

# GPQA 精度评测，跑在已经起好的 vLLM 服务上。
# 服务地址用环境变量覆盖，不用复制 ais_bench 的 config 文件：
#   VLLM_PORT=7969 ./gpqa.sh
# 完整的变量列表和原理见 run_ais_bench.sh 顶部注释。

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$script_dir/run_ais_bench.sh" gpqa_gen_0_shot_cot_chat_prompt "$@"
