#!/usr/bin/env bash
# Collect only the data needed to diagnose Qwen3.8 ModelSlim startup failures.
# It deliberately avoids dumping the full environment because it may contain
# proxy credentials or other secrets.

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
python3 -c '
import importlib.util

for name in ("vllm", "vllm_ascend"):
    spec = importlib.util.find_spec(name)
    print(name + "=" + str(getattr(spec, "origin", None)))
' 2>&1 || true

active_vllm_ascend_root="$(python3 -c '
import importlib.util
import pathlib

spec = importlib.util.find_spec("vllm_ascend")
print(pathlib.Path(spec.origin).parent.parent if spec and spec.origin else "")
' 2>/dev/null || true)"
active_package_root="${active_vllm_ascend_root:+$active_vllm_ascend_root/vllm_ascend}"
active_modelslim_source="${active_package_root:+$active_package_root/quantization/modelslim_config.py}"
active_patch_source="${active_package_root:+$active_package_root/patch/worker/patch_qwen3_5.py}"
echo "active_vllm_ascend_root=$active_vllm_ascend_root"
echo "active_modelslim_source=$active_modelslim_source"
echo "active_qwen3_5_patch_source=$active_patch_source"

if [[ -n "$active_vllm_ascend_root" ]] && \
    [[ -d "$active_vllm_ascend_root/.git" || -f "$active_vllm_ascend_root/.git" ]]; then
    active_vllm_ascend_commit="$(git -C "$active_vllm_ascend_root" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
    echo "active_vllm_ascend_commit=$active_vllm_ascend_commit"
fi

if [[ -z "$checkout_path" ]]; then
    for candidate in \
        "$active_vllm_ascend_root" \
        /home/hajimi/qwen3.8/vllm-ascend/junlin-bugfix-modelslim-qwen35-moe-text \
        /home/hajimi/qwen3.8/vllm-ascend/main \
        /opt/vllm-ascend \
        /home/hajimi/vllm-ascend \
        /home/hajimi/vLLm-ascend; do
        if [[ -n "$candidate" && ( -d "$candidate/.git" || -f "$candidate/.git" ) ]]; then
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
    for pin in .github/vllm-main-verified.commit .github/vllm-release-tag.commit; do
        if [[ -r "$checkout_path/$pin" ]]; then
            printf '%s=' "$pin"
            tr -d '[:space:]' < "$checkout_path/$pin"
            printf '\n'
        fi
    done
    if [[ -n "$active_vllm_ascend_root" ]]; then
        active_matches_checkout=false
        if [[ "$(readlink -f "$active_vllm_ascend_root")" == "$(readlink -f "$checkout_path")" ]]; then
            active_matches_checkout=true
        fi
        echo "active_matches_checkout=$active_matches_checkout"
    fi
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
if [[ -n "$active_patch_source" && -r "$active_patch_source" ]]; then
    checker_args+=(--patch-source "$active_patch_source")
fi
python3 "$script_dir/check_qwen38_modelslim_metadata.py" "${checker_args[@]}"
preflight_status=$?

echo "preflight_exit_code=$preflight_status"
if [[ "$preflight_status" -eq 0 ]]; then
    echo "RESULT: static metadata and RoPE source contracts passed. " \
        "Return this report plus the complete first worker traceback."
else
    echo "RESULT: static contract failed. " \
        "The report identifies the mismatched source, RoPE guard, or checkpoint key."
fi
echo "REPORT_SAVED=$report_path"
exit "$preflight_status"
