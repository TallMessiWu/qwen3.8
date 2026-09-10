#!/usr/bin/env bash
set -euo pipefail

# ais_bench 精度评测的公共入口，gsm8.sh / gpqa.sh / mmmu.sh 都走这里。
# 直接调用时第一个参数是数据集名：./run_ais_bench.sh <dataset> [附加参数...]
#
# 支持的环境变量：
#
#   VLLM_IP         服务 IP 或主机名，默认 localhost
#   VLLM_PORT       服务端口，默认 6969
#   VLLM_URL        完整 URL，给了它就忽略 VLLM_IP / VLLM_PORT
#   MODEL_NAME      模型名，不给则由服务的 /v1/models 自行探测
#   AIS_MODEL_CFG   ais_bench 的模型配置模板，默认 vllm_api_general_chat.py
#
# 数据集名之后的参数原样透传给 ais_bench，例如 --work-dir、--batch-size、--debug。
#
# 为什么走命令行覆盖，而不是改 ais_bench 自带的 configs/models/vllm_api/*.py：
# 那些文件里 import 了 ais_bench 自己的模块，mmengine 的 Config._is_lazy_import
# 因此判定为 lazy import，走 _parse_lazy_import 解析。lazy 模式下文件里的任何函数
# 调用都不会真的执行，只会被包成 LazyObject，一调用就 raise RuntimeError，报成
# TMAN-CFG-001 invalid syntax。所以 os.environ.get("VLLM_PORT", 6969) 这种写法
# 在 config 里必然失败，mmengine 的 {{$ENV:default}} 语法同理（只在非 lazy 路径生效）。
#
# 出路是 ais_bench 的 api_model_args 参数组：--host-ip / --host-port / --url /
# --model-name 等会在 config 加载后覆盖模型字段，且只覆盖 config 里已存在的 key。
# 上面这些变量在 shell 层就展开成命令行参数，config 文件保持原样，一个都不用复制。
# 要固化一套参数时照 ais_bench 自己的做法写薄配置：with read_base() 继承
# vllm_api_general_chat 再改字段，但那样仍然读不了环境变量，端口还是得走命令行。

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
