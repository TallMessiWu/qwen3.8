#!/usr/bin/env bash
# Single-node 8-NPU launcher for a four-layer Qwen3.8-2.4T weight smoke test.
#
# This loads the original BF16 checkpoint. It does not validate the W8A8
# checkpoint, ModelSlim metadata, or multi-node communication.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
layer_filter_dir="${script_dir}/debug/qwen38_checkpoint_layer_filter"
# shellcheck source=hajimi-port.sh
source "$script_dir/hajimi-port.sh"

MODEL_PATH="${MODEL_PATH:-/mnt/share/weight/Qwen3.8-2.4T-A95B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"

if [[ ! -r "$MODEL_PATH/config.json" ]]; then
    echo "ERROR: missing or unreadable $MODEL_PATH/config.json" >&2
    exit 2
fi
if [[ ! -r "$layer_filter_dir/sitecustomize.py" ]]; then
    echo "ERROR: missing checkpoint layer filter at $layer_filter_dir" >&2
    exit 2
fi

active_vllm_ascend_root="$(
    python3 -c 'import importlib.util; from pathlib import Path; spec = importlib.util.find_spec("vllm_ascend"); raise SystemExit("vllm_ascend is not importable") if spec is None or spec.origin is None else print(Path(spec.origin).parent)'
)"
active_qwen_patch="${active_vllm_ascend_root}/patch/worker/patch_qwen3_5.py"
if ! grep -Fq 'if isinstance(self.rotary_emb, AscendMRotaryEmbedding):' "$active_qwen_patch"; then
    echo "ERROR: active vLLM-Ascend lacks the Qwen3.5 text RoPE dispatch fix." >&2
    echo "Active patch: $active_qwen_patch" >&2
    echo "Run: python3 -m pip install --no-deps --no-build-isolation -e /home/hajimi/qwen3.8/vllm-ascend/junlin-bugfix-modelslim-qwen35-moe-text" >&2
    exit 2
fi
echo "QWEN38_ROPE_DISPATCH=GREEN active_patch=$active_qwen_patch"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
# Match the four-layer hf override and skip layers 4+ before safetensors get_tensor().
export QWEN38_CHECKPOINT_LAYER_LIMIT=4
export PYTHONPATH="${layer_filter_dir}${PYTHONPATH:+:${PYTHONPATH}}"

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
    --safetensors-load-strategy lazy
    --dtype bfloat16
    --tensor-parallel-size 8
    --enable-expert-parallel
    --hf-overrides '{"num_hidden_layers":4,"layer_types":["linear_attention","linear_attention","linear_attention","full_attention"]}'
    --max-model-len 2048
    --max-num-seqs 1
    --max-num-batched-tokens 2048
    --gpu-memory-utilization 0.80
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --seed 1024
    --additional-config '{"enable_cpu_binding":true,"enable_flashcomm1":false,"enable_fused_mc2":false}'
)

printf 'Launching four-layer single-node smoke server:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
