#!/usr/bin/env bash
# Single-node 8-NPU launcher for a four-layer Qwen3.8-2.4T weight smoke test.
#
# QUANTIZATION=ascend (default) serves a ModelSlim-quantized checkpoint with
# --quantization ascend and refuses placeholder descriptions whose entries are
# all FLOAT (that means the checkpoint was never actually quantized).
# QUANTIZATION=none serves a plain BF16 checkpoint instead.
# It does not validate multi-node communication.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
layer_filter_dir="${script_dir}/runtime/qwen38_checkpoint_layer_filter"

MODEL_PATH="${MODEL_PATH:-/mnt/share/weight/Qwen3.8-2.4T-A95B-mxfp8}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
VLLM_PORT="${VLLM_PORT:-6969}"
QUANTIZATION="${QUANTIZATION:-ascend}"

if [[ ! -r "$MODEL_PATH/config.json" ]]; then
    echo "ERROR: missing or unreadable $MODEL_PATH/config.json" >&2
    exit 2
fi
if [[ "$QUANTIZATION" == "ascend" ]]; then
    quant_desc="$MODEL_PATH/quant_model_description.json"
    if [[ ! -r "$quant_desc" ]]; then
        echo "ERROR: --quantization ascend requires $quant_desc" >&2
        exit 2
    fi
    if ! grep -Eq '"W[48]A(4|8|16)' "$quant_desc"; then
        echo "ERROR: $quant_desc declares no quantized layers (all FLOAT placeholders)," >&2
        echo "so the checkpoint at $MODEL_PATH is not actually quantized." >&2
        echo "Point MODEL_PATH at a real ModelSlim checkpoint, or run a BF16 smoke test with:" >&2
        echo "  QUANTIZATION=none MODEL_PATH=/mnt/share/weight/Qwen3.8-2.4T-A95B bash $0" >&2
        exit 2
    fi
elif [[ "$QUANTIZATION" != "none" ]]; then
    echo "ERROR: QUANTIZATION must be 'ascend' or 'none'; got '$QUANTIZATION'." >&2
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
if ! grep -Fq '_uses_multimodal_rope' "$active_qwen_patch"; then
    echo "ERROR: active vLLM-Ascend lacks the Qwen3.5 text RoPE dispatch fix." >&2
    echo "Active patch: $active_qwen_patch" >&2
    echo "Run: python3 -m pip install --no-deps --no-build-isolation -e /home/hajimi/qwen3.8/vllm-ascend/feat-qfa-mxfp8-attn" >&2
    exit 2
fi
echo "QWEN38_ROPE_DISPATCH=GREEN active_patch=$active_qwen_patch"
if [[ "$QUANTIZATION" == "ascend" ]]; then
    active_modelslim_config="${active_vllm_ascend_root}/quantization/modelslim_config.py"
    if ! grep -Fq '"qwen3_5_moe_text"' "$active_modelslim_config"; then
        echo "ERROR: active vLLM-Ascend lacks the qwen3_5_moe_text ModelSlim packed mapping." >&2
        echo "Active config: $active_modelslim_config" >&2
        echo "Pull the feat/qfa-mxfp8-attn branch (rebased on upstream/main with the PR14238 mapping) and retry." >&2
        exit 2
    fi
    echo "QWEN38_MODELSLIM_MAPPING=GREEN active_config=$active_modelslim_config"
fi

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
if [[ "$QUANTIZATION" == "ascend" ]]; then
    cmd+=(--quantization ascend)
fi

printf 'Launching four-layer single-node smoke server:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
