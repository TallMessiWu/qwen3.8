#!/usr/bin/env bash
# Single-node 8-NPU launcher for a four-layer Qwen3.8-2.4T smoke test.
#
# This uses random BF16 weights and does not validate the W8A8 checkpoint,
# ModelSlim metadata, or multi-node communication.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/share/weight/Qwen3.8-2.4T-A95B-w8a8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
VLLM_PORT="${VLLM_PORT:-8000}"

if [[ ! -r "$MODEL_PATH/config.json" ]]; then
    echo "ERROR: missing or unreadable $MODEL_PATH/config.json" >&2
    exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"

IFS=',' read -r -a visible_devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
if [[ "${#visible_devices[@]}" -ne 8 ]]; then
    echo "ERROR: this launcher requires 8 visible NPUs; got ${#visible_devices[@]}." >&2
    exit 2
fi

cmd=(
    vllm serve "$MODEL_PATH"
    --host 0.0.0.0
    --port "$VLLM_PORT"
    --served-model-name qwen3.8-smoke
    --tokenizer "$TOKENIZER_PATH"
    --trust-remote-code
    --load-format dummy
    --dtype bfloat16
    --tensor-parallel-size 8
    --enable-expert-parallel
    --hf-overrides '{"num_hidden_layers":4,"layer_types":["linear_attention","linear_attention","linear_attention","full_attention"]}'
    --max-model-len 2048
    --max-num-seqs 1
    --max-num-batched-tokens 2048
    --gpu-memory-utilization 0.80
    --enforce-eager
    --seed 1024
    --additional-config '{"enable_cpu_binding":true,"enable_flashcomm1":false,"enable_fused_mc2":false}'
)

printf 'Launching four-layer single-node smoke server:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
