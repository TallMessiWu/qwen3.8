#!/usr/bin/env bash
# Collect only the data needed to distinguish the Qwen3.8 ModelSlim KeyError
# hypotheses.  It deliberately avoids dumping the full environment because it
# may contain proxy credentials or other secrets.

set -uo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
    echo "Usage: bash $0 MODEL_PATH [VLLM_ASCEND_CHECKOUT]" >&2
    exit 2
fi

model_path="$1"
checkout_path="${2:-}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_dir="$script_dir/output"
mkdir -p "$output_dir"
report_path="${REPORT_PATH:-$output_dir/qwen38-modelslim-$(hostname)-$(date +%Y%m%d-%H%M%S).log}"

exec > >(tee "$report_path") 2>&1

echo "report_path=$report_path"
echo "timestamp=$(date --iso-8601=seconds 2>/dev/null || date)"
echo "hostname=$(hostname)"
echo "kernel=$(uname -a)"
echo "python=$(python3 --version 2>&1)"

echo "[package versions]"
python3 -m pip show vllm vllm-ascend torch torch-npu 2>&1 || true

echo "[active Python package paths]"
python3 -c 'import importlib.util; [print(name + "=" + str(getattr(importlib.util.find_spec(name), "origin", None))) for name in ("vllm", "vllm_ascend")]' 2>&1 || true

active_modelslim_source="$(python3 -c 'import importlib.util, pathlib; spec=importlib.util.find_spec("vllm_ascend"); print(pathlib.Path(spec.origin).parent / "quantization" / "modelslim_config.py" if spec and spec.origin else "")' 2>/dev/null || true)"
echo "active_modelslim_source=$active_modelslim_source"

if [[ -z "$checkout_path" ]]; then
    for candidate in /opt/vllm-ascend /home/hajimi/vllm-ascend /home/hajimi/vLLm-ascend; do
        if [[ -d "$candidate/.git" || -f "$candidate/.git" ]]; then
            checkout_path="$candidate"
            break
        fi
    done
fi

echo "[source checkout]"
if [[ -n "$checkout_path" && ( -d "$checkout_path/.git" || -f "$checkout_path/.git" ) ]]; then
    echo "checkout_path=$checkout_path"
    git -C "$checkout_path" rev-parse HEAD 2>&1 || true
    git -C "$checkout_path" status --short 2>&1 || true
else
    echo "checkout_path=NOT_FOUND"
fi

echo "[selected distributed environment]"
for name in \
    ASCEND_RT_VISIBLE_DEVICES HCCL_IF_IP GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME \
    HCCL_SOCKET_IFNAME HCCL_BUFFSIZE HCCL_BUFFSIZE_EP HCCL_INTRA_PCIE_ENABLE \
    HCCL_INTRA_ROCE_ENABLE VLLM_ASCEND_ENABLE_FUSED_MC2; do
    printf '%s=%s\n' "$name" "${!name-<UNSET>}"
done

echo "[network interface]"
if command -v ip >/dev/null 2>&1; then
    ip -o -4 addr show 2>&1 || true
else
    echo "ip command not found"
fi

echo "[NPU inventory]"
if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info 2>&1 || true
else
    echo "npu-smi command not found"
fi

echo "[ModelSlim preflight]"
checker_args=("$model_path")
if [[ -n "$active_modelslim_source" && -r "$active_modelslim_source" ]]; then
    checker_args+=(--modelslim-source "$active_modelslim_source")
fi
python3 "$script_dir/check_qwen38_modelslim_metadata.py" "${checker_args[@]}"
preflight_status=$?

echo "preflight_exit_code=$preflight_status"
if [[ "$preflight_status" -eq 0 ]]; then
    echo "RESULT: static metadata contract passed. Return this report plus the complete first worker traceback."
else
    echo "RESULT: static metadata contract failed. This report should identify the mismatched source or checkpoint key."
fi
echo "REPORT_SAVED=$report_path"
exit "$preflight_status"
