#!/usr/bin/env bash
set -euo pipefail

# MMMU 精度评测，跑在已经起好的 vLLM 服务上。多模态数据集，服务侧要能收图。
#
# 用环境变量指定服务，不用复制 ais_bench 的 config 文件（为什么不能在 config 里
# 读环境变量、这些开关又是怎么生效的，见 run_ais_bench.sh 顶部注释）：
#
#   VLLM_IP         服务 IP 或主机名，默认 localhost
#   VLLM_PORT       服务端口，默认 6969
#   VLLM_URL        完整 URL，给了它就忽略 VLLM_IP / VLLM_PORT
#   MODEL_NAME      模型名，不给则由服务的 /v1/models 自行探测
#   AIS_MODEL_CFG   ais_bench 的模型配置模板，默认 vllm_api_general_chat.py
#
# 脚本名之后的参数原样透传给 ais_bench，例如 --work-dir、--batch-size、--debug。
#
#   ./mmmu.sh
#   VLLM_PORT=7969 ./mmmu.sh                        # 同机的另一个服务
#   VLLM_IP=10.0.0.5 VLLM_PORT=8000 ./mmmu.sh       # 别的机器上的服务
#   VLLM_URL=http://gw.example/prefix/ ./mmmu.sh    # 带路径的网关
#   MODEL_NAME=qwen3.8 ./mmmu.sh --batch-size 16

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$script_dir/run_ais_bench.sh" mmmu_gen "$@"
