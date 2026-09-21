#!/usr/bin/env bash
set -euo pipefail

# ais_bench 精度评测的公共入口，gsm8k.sh / gpqa.sh / mmmu.sh 都走这里。
# 直接调用时第一个参数是数据集名：./run_ais_bench.sh <dataset> [附加参数...]
#
# 支持的环境变量：
#
#   VLLM_IP         服务 IP 或主机名，默认 localhost
#   VLLM_PORT       服务端口，默认 6969
#   VLLM_URL        完整 URL，给了它就忽略 VLLM_IP / VLLM_PORT
#   MODEL_NAME      模型名，不给则由服务的 /v1/models 自行探测
#   AIS_MODEL_CFG   ais_bench 自带的模型配置模板，默认 vllm_api_general_chat
#   PREFLIGHT       设为 0 跳过开跑前的预检，默认开
#
# 数据集名之后的参数原样透传给 ais_bench，例如 --work-dir、--batch-size、--debug。
#
# 预检（ais_bench_preflight.py）为什么默认开：ais_bench 对失败的请求只留 HTTP 状态
# 短语，响应体直接丢掉，而 vLLM 把拒绝的原因写在响应体里。服务端一旦拒了请求，这边
# 只看得到 "Bad Request" 两个词，warmup 全败后它还会接着跑评测阶段、汇总出一张空表、
# 以退出码 0 收场。预检照 ais_bench 的拼法先发同一个请求，把响应体打出来，RED 就
# 不往下跑。直接敲 ais_bench 命令绕过本脚本，就同时绕过了地址覆盖和这一步。
#
# 为什么不把地址写进 ais_bench 自带的 configs/models/vllm_api/*.py：
# 那些文件 import 了 ais_bench 自己的模块，mmengine 的 Config._is_lazy_import 因此
# 判定为 lazy import，走 _parse_lazy_import 解析。lazy 模式下文件里的任何函数调用都
# 不会真的执行，只会被包成 LazyObject，一调用就 raise RuntimeError，报成
# TMAN-CFG-001 invalid syntax。所以 os.environ.get("VLLM_PORT", 6969) 这种写法在
# config 里必然失败，mmengine 的 {{$ENV:default}} 语法同理（只在非 lazy 路径生效）。
#
# 于是按 ais_bench 的版本分两条路，脚本自己探测，不用手工切换：
#
#   新版（2026-08-27 的 #492 起，tag v3.1-20260827-master 及以后）带 api_model_args
#   参数组，--host-ip / --host-port / --url / --model-name 在 config 加载后覆盖字段，
#   只覆盖 config 里已存在的 key。直接传参即可。
#
#   老版没有那组参数。改为用 gen_ais_bench_model_cfg.py 读本机装的那份模板，替换掉
#   地址字段，生成到 .ais_bench_configs/models/ 下，再用 --config-dir 指过去。
#   ais_bench 找 models/datasets 时自定义目录优先、自带目录兜底，所以数据集仍走它
#   自己那份。生成的文件按端点命名，同时跑多个服务不会互相覆盖。

if [[ $# -lt 1 ]]; then
    echo "用法: $0 <dataset> [ais_bench 附加参数...]" >&2
    exit 2
fi

dataset="$1"
shift

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v ais_bench >/dev/null 2>&1; then
    echo "RED: 找不到 ais_bench，确认它在 PATH 里" >&2
    exit 127
fi

template="${AIS_MODEL_CFG:-vllm_api_general_chat}"
template="${template%.py}"

args=(--datasets "$dataset" --dump-eval-details)

if [[ -n "${VLLM_URL:-}" ]]; then
    endpoint="$VLLM_URL"
else
    endpoint="${VLLM_IP:-localhost}:${VLLM_PORT:-6969}"
fi

# 用 ais_bench 自己的解释器跑两个辅助脚本，否则可能 import 不到它。判据就是能不能
# import 到，因为 shebang 可能是 /usr/bin/env python3 这种，第一个词并不是解释器本身。
ais_python="$(head -1 "$(command -v ais_bench)" | sed 's|^#!||' | awk '{print $1}')"
if [[ ! -x "$ais_python" ]] || ! "$ais_python" -c "import ais_bench" >/dev/null 2>&1; then
    ais_python=python3
fi

endpoint_args=()
if [[ -n "${VLLM_URL:-}" ]]; then
    endpoint_args+=(--url "$VLLM_URL")
else
    endpoint_args+=(--host-ip "${VLLM_IP:-localhost}" --host-port "${VLLM_PORT:-6969}")
fi
if [[ -n "${MODEL_NAME:-}" ]]; then
    endpoint_args+=(--model-name "$MODEL_NAME")
fi

if ais_bench --help 2>&1 | grep -q -- '--host-ip'; then
    mode="cli-override"
    args+=(--models "$template.py" "${endpoint_args[@]}")
    preflight_cfg=(--template "$template")
else
    mode="generated-config"
    cfg_dir="$script_dir/.ais_bench_configs"
    cfg_name="qwen38_$(printf '%s' "$endpoint" | tr -c 'A-Za-z0-9' '_')"

    generated="$("$ais_python" "$script_dir/gen_ais_bench_model_cfg.py" \
        --out-dir "$cfg_dir" --name "$cfg_name" --template "$template" "${endpoint_args[@]}")"
    echo "[ais_bench] 本机 ais_bench 不支持 --host-ip，已生成 $generated" >&2
    args+=(--models "$cfg_name" --config-dir "$cfg_dir")
    preflight_cfg=(--config "$generated")
fi

echo "[ais_bench] dataset=$dataset endpoint=$endpoint mode=$mode" >&2

if [[ "${PREFLIGHT:-1}" != "0" ]]; then
    # "$@" 也交给它：新版 ais_bench 的 --max-out-len / --generation-kwargs 会改请求体。
    if ! "$ais_python" "$script_dir/ais_bench_preflight.py" \
            "${endpoint_args[@]}" "${preflight_cfg[@]}" --dataset "$dataset" -- "$@"; then
        echo "[ais_bench] 预检 RED，没有启动评测。确认是误报再用 PREFLIGHT=0 跳过。" >&2
        exit 1
    fi
fi

exec ais_bench "${args[@]}" "$@"
