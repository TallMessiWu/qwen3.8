#!/usr/bin/env bash
# Single-node Qwen3.8-27B-MXFP8 baseline, retaining the user's host tuning and
# optional npu-cleaner workflow while fixing the empty default device list.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cleaner_path="$script_dir/npu-cleaner.sh"

if [[ "$#" -gt 0 ]]; then
    old_ifs="$IFS"
    IFS=','
    visible_devices="$*"
    IFS="$old_ifs"
else
    visible_devices="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
fi
export ASCEND_RT_VISIBLE_DEVICES="$visible_devices"
IFS=',' read -r -a device_args <<< "$ASCEND_RT_VISIBLE_DEVICES"

if [[ -x "$cleaner_path" ]]; then
    echo "Cleaning NPU devices: $ASCEND_RT_VISIBLE_DEVICES"
    "$cleaner_path" "${device_args[@]}"
    sleep 1
else
    echo "WARNING: executable cleaner not found at $cleaner_path; continuing without cleanup." >&2
fi

export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/lib64"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export HCCL_IF_IP="${HCCL_IF_IP:-127.0.0.1}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"

if [[ "${ENABLE_HOST_TUNING:-1}" == "1" ]]; then
    for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [[ -w "$governor" ]] && printf 'performance\n' > "$governor"
    done
    if command -v sysctl >/dev/null 2>&1; then
        sysctl -w vm.swappiness=0 || true
        sysctl -w kernel.numa_balancing=0 || true
        sysctl -w kernel.sched_migration_cost_ns=50000 || true
    fi
fi

MODEL_PATH="${MODEL_PATH:-/mnt/share/weight/Qwen3.8-27B-mxfp8}"
VLLM_PORT="${VLLM_PORT:-6969}"
MODEL_NAME="${MODEL_NAME:-qwen3.8}"
# Lower this to leave the aclgraph pool room. Capturing QFA needs roughly 1.28x
# one attention layer's bf16 KV cache in transient tensors (measured by the
# POOL-MEM case of scripts/debug/test_qfa_graph_capture_npu.py), which 0.95
# leaves no space for. Both the cache and that requirement shrink together, so
# giving back a few points frees more than it costs. Goes away with an MXFP8
# cache, when the per-step quantization does.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"

# QFA=1 runs the causal full-attention path on the vendored QuantFlashAttn
# instead of FIA (junlin-qfa branch, VLLM_ASCEND_ENABLE_QFA). The KV cache is
# still bf16: K/V are quantized to MXFP8 on every step, so this saves no memory
# yet -- it is the probe, not the destination. Prefix caching comes off because
# the MXFP8 cache it is heading for shares E8M0 scales across tokens, which
# shared blocks cannot track; a QFA-vs-FIA comparison therefore has to pass
# --no-enable-prefix-caching to the baseline run as well, or the two differ by
# more than the attention op.
# GRAPH=0 turns off aclgraph capture (eager). QFA=1 covers prefill and eager
# decode only: on this branch the captured decode graph still replays FIA, for
# the target model as well as for the MTP drafter. Getting QuantFlashAttn into
# the graph is the open subgoal -- the junlin-qfa-graph branch is where that is
# being chased, and it crashes. MTP=0 turns off speculative decoding. Both
# default to on.
qfa_args=()
if [[ "${QFA:-0}" == "1" ]]; then
    export VLLM_ASCEND_ENABLE_QFA=1
    echo "QFA on: QuantFlashAttn for causal attention; KV cache still bf16 (quantized per step)." >&2
fi
# Also settable on its own, so a QFA-vs-FIA comparison can hold it constant.
if [[ "${QFA:-0}" == "1" || "${NO_PREFIX_CACHE:-0}" == "1" ]]; then
    qfa_args+=(--no-enable-prefix-caching)
    echo "prefix caching disabled." >&2
fi

compilation_config='{"cudagraph_capture_sizes":[1,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128],"cudagraph_mode":"FULL_DECODE_ONLY"}'
if [[ "${GRAPH:-1}" == "0" ]]; then
    compilation_config='{"cudagraph_mode":"NONE"}'
    echo "aclgraph capture disabled (GRAPH=0)." >&2
fi

spec_args=(--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}')
if [[ "${MTP:-1}" == "0" ]]; then
    spec_args=()
    echo "MTP speculative decoding disabled (MTP=0)." >&2
fi

exec vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --data-parallel-size 1 \
    --tensor-parallel-size 1 \
    --max-model-len 133120 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 32 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --reasoning-parser qwen3 \
    --compilation-config "$compilation_config" \
    "${spec_args[@]}" \
    --trust-remote-code \
    --async-scheduling \
    --allowed-local-media-path / \
    --mm-processor-cache-gb 0 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --additional-config '{"enable_cpu_binding":true}' \
    "${qfa_args[@]}"
