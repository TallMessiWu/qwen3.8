#!/usr/bin/env bash
# Single-node Qwen3.5-397B launcher: 8 NPUs, TP8 with expert parallelism.
#
# Same shape as 27B.sh -- the QFA / C8 / MTP / GRAPH switches behave identically
# and mean the same thing -- but the model is a MoE that needs all eight cards,
# so this adds expert parallelism, the HCCL buffer sizing the EP all-to-all
# wants, and lazy safetensors loading.

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

# TP has to match the visible device count exactly: vLLM would otherwise fail
# deep inside worker startup with a message that does not name the cause.
TP_SIZE="${TP_SIZE:-8}"
if ! [[ "$TP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "TP_SIZE must be a positive integer, got '$TP_SIZE'." >&2
    exit 2
fi
if [[ "${#device_args[@]}" -ne "$TP_SIZE" ]]; then
    echo "TP_SIZE=$TP_SIZE but ASCEND_RT_VISIBLE_DEVICES exposes ${#device_args[@]} device(s)." >&2
    exit 2
fi

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
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export HCCL_IF_IP="${HCCL_IF_IP:-127.0.0.1}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
# EP moves every token's hidden state between cards twice a MoE layer, so the
# EP buffer is sized well above the default. Both stay inside one node here,
# hence PCIe on and RoCE off.
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export HCCL_BUFFSIZE_EP="${HCCL_BUFFSIZE_EP:-2048}"
export HCCL_INTRA_PCIE_ENABLE="${HCCL_INTRA_PCIE_ENABLE:-1}"
export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-0}"

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

MODEL_PATH="${MODEL_PATH:-/mnt/share/weight/qwen3.5-397b-w4a4_multi}"
VLLM_PORT="${VLLM_PORT:-6969}"
MODEL_NAME="${MODEL_NAME:-qwen3.8}"
# Weights are ~4 bits, so eight cards hold them with room to spare -- unlike
# 27B and 2.4T, the memory ceiling here is not what limits the run. Start
# conservative and raise it if the KV cache comes up short: the aclgraph pool is
# captured after the KV allocation and is not counted in the profiling result,
# so whatever this leaves unclaimed is what the graphs get.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

# EP=1 spreads the experts across the eight TP ranks instead of replicating
# them. Off is only useful as a comparison point; a 397B MoE on one node wants
# it on.
ep_args=()
if [[ "${EP:-1}" == "1" ]]; then
    ep_args+=(--enable-expert-parallel)
else
    echo "expert parallelism disabled (EP=0)." >&2
fi
# MegaMoe's fused dispatch/combine. vllm-ascend defaults it off and silently
# turns it back off for model configs it does not cover, so leave it off while
# bringing a checkpoint up and flip it only to measure it.
FUSED_MC2="${FUSED_MC2:-0}"
if [[ "$FUSED_MC2" != "0" && "$FUSED_MC2" != "1" ]]; then
    echo "FUSED_MC2 must be 0 or 1, got '$FUSED_MC2'." >&2
    exit 2
fi
export VLLM_ASCEND_ENABLE_FUSED_MC2="$FUSED_MC2"

# QFA=1 runs the causal full-attention path on the vendored QuantFlashAttn
# instead of FIA (VLLM_ASCEND_ENABLE_QFA). It only takes effect on a C8-MXFP KV
# cache -- the checkpoint's quant description has to select MXFP8 for the KV
# cache and carry a per-layer V scale -- because that cache is the only thing
# QFA can read. With a bf16 cache the flag is ignored and the run is the FIA
# baseline. Qwen3.5 is a 3+1 hybrid, so only the full-attention quarter of the
# layers has a KV cache at all; the GDN layers are unaffected either way.
# Prefix caching comes off because the MXFP8 cache shares E8M0 scales across
# tokens, which shared blocks cannot track; a QFA-vs-FIA comparison therefore
# has to pass --no-enable-prefix-caching to the baseline run as well, or the
# two differ by more than the attention op.
# GRAPH=0 turns off aclgraph capture (eager); it defaults to on.
#
# C8=0 forces the MXFP8 KV cache off (VLLM_ASCEND_DISABLE_C8_MXFP), serving a
# quantized checkpoint on a bf16 cache through plain FIA. That is the only
# baseline an end-to-end accuracy or memory comparison can use: with the C8
# cache on, the FIA path hits EZ0010 at head_dim 256, so "MXFP8 cache + FIA"
# does not exist as a configuration. QFA cannot read a bf16 cache, so C8=0
# implies QFA off. C8=1 (default) honours whatever the checkpoint asks for.
qfa_args=()
if [[ "${C8:-1}" == "0" ]]; then
    export VLLM_ASCEND_DISABLE_C8_MXFP=1
    if [[ "${QFA:-0}" == "1" ]]; then
        echo "C8=0 forces a bf16 KV cache, which QFA cannot read; QFA=1 has no effect." >&2
    fi
    echo "C8 MXFP8 KV cache disabled: bf16 cache served by FIA." >&2
elif [[ "${QFA:-0}" == "1" ]]; then
    export VLLM_ASCEND_ENABLE_QFA=1
    echo "QFA on: QuantFlashAttn reads the MXFP8 KV cache directly." >&2
fi
# Also settable on its own, so a QFA-vs-FIA comparison can hold it constant.
# C8=0 turns it off too, so the baseline run matches the C8 run here.
if [[ "${QFA:-0}" == "1" || "${C8:-1}" == "0" || "${NO_PREFIX_CACHE:-0}" == "1" ]]; then
    qfa_args+=(--no-enable-prefix-caching)
    echo "prefix caching disabled." >&2
fi

# MAX_MODEL_LEN is what _qfa_max_seqlen_kv reads, and the captured
# QuantFlashAttn bakes that constant in -- AdjustSinnerAndSouter tiles on it.
# MAX_NUM_SEQS bounds the batch, and with it the plan's per-core split and the
# graph plan below. It is lower than 27B's because the graph count follows it
# one-for-one, and capturing that many graphs across eight ranks of a 397B
# model costs real startup time.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
if ! [[ "$MAX_NUM_SEQS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_NUM_SEQS must be a positive integer, got '$MAX_NUM_SEQS'." >&2
    exit 1
fi

# MTP is num_speculative_tokens, not a flag: MTP=3 proposes three draft tokens
# per step, MTP=0 turns speculative decoding off. It sets the width of a decode
# step -- one accepted token plus MTP drafts -- which is what the graph plan
# below is built on. Default off: this checkpoint is still being brought up,
# and speculative decoding is one more thing to rule out when it will not load.
MTP="${MTP:-0}"
if ! [[ "$MTP" =~ ^[0-9]+$ ]]; then
    echo "MTP must be a non-negative integer (num_speculative_tokens), got '$MTP'." >&2
    exit 1
fi
if [[ "$MTP" == "0" ]]; then
    spec_args=()
    decode_query_len=1
    echo "MTP speculative decoding disabled (MTP=0)." >&2
else
    spec_args=(--speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$MTP}")
    decode_query_len=$((MTP + 1))
fi

# A uniform decode step is num_reqs * decode_query_len tokens, and
# CudagraphDispatcher.dispatch drops to eager the moment that exceeds the
# largest captured size -- so the plan has to track both MAX_NUM_SEQS and MTP
# or the top of the batch range silently runs ungraphed. Hence one size per
# request count, stepping by decode_query_len up to the full batch. Sizes that
# are not multiples of decode_query_len are pointless: vLLM's
# adjust_cudagraph_sizes_for_spec_decode round_up's every entry to that
# multiple. CAPTURE_SIZES overrides the whole list for single-variable
# experiments, as does CUDAGRAPH_MODE for keeping attention out of the graph.
default_capture_sizes=""
for ((n = 1; n <= MAX_NUM_SEQS; n++)); do
    default_capture_sizes+="${default_capture_sizes:+,}$((n * decode_query_len))"
done
capture_sizes="${CAPTURE_SIZES:-$default_capture_sizes}"
cudagraph_mode="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
compilation_config="{\"cudagraph_capture_sizes\":[$capture_sizes],\"cudagraph_mode\":\"$cudagraph_mode\"}"
if [[ "${GRAPH:-1}" == "0" ]]; then
    compilation_config='{"cudagraph_mode":"NONE"}'
    echo "aclgraph capture disabled (GRAPH=0)." >&2
else
    echo "aclgraph: mode=$cudagraph_mode query_len=$decode_query_len" \
         "sizes=[$capture_sizes]" >&2
fi

# The Qwen3 chat template only injects the empty <think></think> block when
# enable_thinking is explicitly false, so an absent kwarg already means
# thinking on. Passing it makes that the server's own default instead of the
# template's, which is what --reasoning-parser qwen3 assumes anyway, and gives
# THINKING=0 a lever that does not need every client to send
# chat_template_kwargs. A request that sends its own enable_thinking still
# wins -- the CLI value is only a default to merge under it.
enable_thinking=true
if [[ "${THINKING:-1}" == "0" ]]; then
    enable_thinking=false
    echo "thinking disabled (THINKING=0)." >&2
fi

if [[ ! -r "$MODEL_PATH/config.json" ]]; then
    echo "ERROR: missing or unreadable $MODEL_PATH/config.json" >&2
    exit 2
fi
if [[ ! -r "$MODEL_PATH/quant_model_description.json" ]]; then
    echo "ERROR: --quantization ascend requires $MODEL_PATH/quant_model_description.json" >&2
    exit 2
fi

# The multimodal flags are rejected outright by a text-only model, and this
# checkpoint family ships in both shapes, so take the answer from config.json
# rather than from the directory name.
mm_args=()
if grep -q '"vision_config"' "$MODEL_PATH/config.json"; then
    mm_args+=(
        --allowed-local-media-path /
        --mm-processor-cache-gb 0
        --mm-encoder-tp-mode data
        --mm-processor-cache-type shm
    )
    echo "config.json declares a vision tower: multimodal serving on." >&2
fi

# Profiling is armed, not on: NPUWorker.profile() only builds the
# TorchNPUProfilerWrapper when /start_profile is posted, so carrying the config
# costs nothing until somebody asks for a trace. The path is relative and
# _validate_profiler_config abspath's it against the CWD, so launching from a
# different directory puts the traces somewhere else. with_stack is off because
# the Python stacks dwarf the op timeline that a kernel-level look is after.
exec vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --quantization ascend \
    --safetensors-load-strategy lazy \
    --dtype bfloat16 \
    --data-parallel-size 1 \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 16384 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs "{\"enable_thinking\": $enable_thinking}" \
    --compilation-config "$compilation_config" \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./profiling", "torch_profiler_with_stack": false}' \
    "${spec_args[@]}" \
    "${ep_args[@]}" \
    --trust-remote-code \
    --async-scheduling \
    "${mm_args[@]}" \
    --additional-config "{\"enable_cpu_binding\":true,\"enable_fused_mc2\":$FUSED_MC2}" \
    "${qfa_args[@]}"
