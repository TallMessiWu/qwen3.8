#!/usr/bin/env bash
# Common launcher for Qwen3.8-2.4T-A95B-W8A8 on four 8-NPU nodes.
#
# Machine-specific wrappers are 2.4T-0.sh through 2.4T-3.sh.  Keep MTP off
# until the base model has loaded successfully; enable it later with
# ENABLE_MTP=1.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${NODE_RANK:?NODE_RANK must be set to 0, 1, 2, or 3}"
: "${LOCAL_IP:?LOCAL_IP must be set to the IP owned by NIC_NAME}"
: "${NODE0_IP:?NODE0_IP must be set to the node-0 IP}"

case "$NODE_RANK" in
    0|1|2|3) ;;
    *)
        echo "ERROR: NODE_RANK must be 0, 1, 2, or 3; got '$NODE_RANK'." >&2
        exit 2
        ;;
esac

MODEL_PATH="${MODEL_PATH:-/mnt/share/weight/Qwen3.8-2.4T-A95B-w8a8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
NIC_NAME="${NIC_NAME:-enp35s0f2}"
VLLM_PORT="${VLLM_PORT:-8000}"
DP_RPC_PORT="${DP_RPC_PORT:-13389}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
ENABLE_MTP="${ENABLE_MTP:-0}"
ENABLE_FUSED_MC2="${ENABLE_FUSED_MC2:-1}"

if ! [[ "$TP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TP_SIZE must be a positive integer; got '$TP_SIZE'." >&2
    exit 2
fi
if [[ "$ENABLE_FUSED_MC2" != "0" && "$ENABLE_FUSED_MC2" != "1" ]]; then
    echo "ERROR: ENABLE_FUSED_MC2 must be 0 or 1; got '$ENABLE_FUSED_MC2'." >&2
    exit 2
fi

if [[ ! -r "$MODEL_PATH/config.json" ]]; then
    echo "ERROR: missing or unreadable $MODEL_PATH/config.json" >&2
    exit 2
fi
if [[ ! -r "$MODEL_PATH/quant_model_description.json" ]]; then
    echo "ERROR: --quantization ascend requires $MODEL_PATH/quant_model_description.json" >&2
    exit 2
fi

if command -v ip >/dev/null 2>&1; then
    if ! ip -o -4 addr show dev "$NIC_NAME" 2>/dev/null | grep -Fq " $LOCAL_IP/"; then
        echo "ERROR: NIC_NAME=$NIC_NAME does not own LOCAL_IP=$LOCAL_IP inside this container." >&2
        exit 2
    fi
fi

export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3000}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export HCCL_BUFFSIZE_EP="${HCCL_BUFFSIZE_EP:-2048}"
export HCCL_INTRA_PCIE_ENABLE="${HCCL_INTRA_PCIE_ENABLE:-1}"
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-0}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export VLLM_ASCEND_ENABLE_FUSED_MC2="$ENABLE_FUSED_MC2"

IFS=',' read -r -a visible_devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
if [[ "${#visible_devices[@]}" -ne "$TP_SIZE" ]]; then
    echo "ERROR: TP_SIZE=$TP_SIZE but ASCEND_RT_VISIBLE_DEVICES exposes ${#visible_devices[@]} devices." >&2
    exit 2
fi

cmd=(
    vllm serve "$MODEL_PATH"
    --host 0.0.0.0
    --port "$VLLM_PORT"
    --served-model-name qwen3.8
    --tokenizer "$TOKENIZER_PATH"
    --trust-remote-code
    --quantization ascend
    --safetensors-load-strategy lazy
    --tensor-parallel-size "$TP_SIZE"
    --data-parallel-size "$DP_SIZE"
    --data-parallel-size-local 1
    --data-parallel-address "$NODE0_IP"
    --data-parallel-rpc-port "$DP_RPC_PORT"
    --enable-expert-parallel
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --seed 1024
    --enable-prefix-caching
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --additional-config "{\"enable_cpu_binding\":true,\"enable_flashcomm1\":false,\"enable_fused_mc2\":$ENABLE_FUSED_MC2}"
)

if [[ "$NODE_RANK" != "0" ]]; then
    cmd+=(--headless --data-parallel-start-rank "$NODE_RANK")
fi

if [[ "$ENABLE_MTP" == "1" ]]; then
    cmd+=(--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}')
elif [[ "$ENABLE_MTP" != "0" ]]; then
    echo "ERROR: ENABLE_MTP must be 0 or 1; got '$ENABLE_MTP'." >&2
    exit 2
fi

bash "$script_dir/npu-cleaner.sh" all

printf 'Launching node %s:' "$NODE_RANK"
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
