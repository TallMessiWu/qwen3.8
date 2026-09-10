#!/usr/bin/env bash
set -euo pipefail

# ais_bench 精度评测的公共入口，gsm8.sh / gpqa.sh 都走这里。
#
# 为什么不去改 ais_bench 自带的 configs/models/vllm_api/*.py：
# 那些文件里 import 了 ais_bench 自己的模块，mmengine 的 Config._is_lazy_import
# 因此判定为 lazy import，走 _parse_lazy_import 解析。lazy 模式下文件里的任何函数
# 调用都不会真的执行，只会被包成 LazyObject，一调用就 raise RuntimeError，报成
# TMAN-CFG-001 invalid syntax。所以 os.environ.get("VLLM_PORT", 6969) 这种写法
# 在 config 里必然失败，mmengine 的 {{$ENV:default}} 语法同理（只在非 lazy 路径生效）。
#
# 出路是 ais_bench 的 api_model_args 参数组：--host-ip / --host-port / --url /
# --model-name 等会在 config 加载后覆盖模型字段，且只覆盖 config 里已存在的 key。
# 变量在 shell 层就展开成命令行参数，config 文件保持原样，一个都不用复制。
#
#   VLLM_PORT=7969 ./gsm8.sh                     # 同机的另一个服务
#   VLLM_IP=10.0.0.5 VLLM_PORT=8000 ./gsm8.sh    # 别的机器上的服务
#   VLLM_URL=http://gw.example/prefix/ ./gsm8.sh # 带路径的网关，此时 IP/PORT 被忽略
#   MODEL_NAME=qwen3.8 ./gsm8.sh                 # 不设则由服务的 /v1/models 自行探测
#   AIS_MODEL_CFG=vllm_api_stream_chat.py ./gsm8.sh   # 换一份 ais_bench 自带模板

if [[ $# -lt 1 ]]; then
    echo "用法: $0 <dataset> [ais_bench 附加参数...]" >&2
    exit 2
fi

dataset="$1"
shift

args=(
    --models "${AIS_MODEL_CFG:-vllm_api_general_chat.py}"
    --datasets "$dataset"
    --dump-eval-details
)

# url 非空时 ais_bench 会忽略 host_ip/host_port，所以这两条路互斥。
if [[ -n "${VLLM_URL:-}" ]]; then
    args+=(--url "$VLLM_URL")
    endpoint="$VLLM_URL"
else
    endpoint="${VLLM_IP:-localhost}:${VLLM_PORT:-6969}"
    args+=(--host-ip "${VLLM_IP:-localhost}" --host-port "${VLLM_PORT:-6969}")
fi

# 留空则不传，让 ais_bench 去 /v1/models 探测服务实际加载的模型名。
if [[ -n "${MODEL_NAME:-}" ]]; then
    args+=(--model-name "$MODEL_NAME")
fi

echo "[ais_bench] dataset=$dataset endpoint=$endpoint" >&2
exec ais_bench "${args[@]}" "$@"
