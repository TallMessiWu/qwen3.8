#!/usr/bin/env bash
# Single-node Qwen3.5-397B launcher. Deliberately identical to 27B.sh apart
# from the checkpoint, TP8 and expert parallelism, so every switch means the
# same thing in both and the two can be read side by side during an experiment.

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
# EP-only, which is why 27B.sh does not carry these. Expert parallelism moves
# every token's hidden state between ranks twice per MoE layer, and the default
# HCCL buffer is not sized for that -- the 2.4T launcher needed these same
# values. All eight ranks sit in one node here, hence PCIe on and RoCE off.
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

# EP=0 drops expert parallelism. It is the only way to take the MoE
# comm-method choice out of the picture: select_moe_comm_method returns
# all-gather outright without EP, so the MC2/ALLTOALL switch at
# mc2_tokens_capacity -- which equals the largest captured graph size -- never
# fires. A 397B MoE wants EP on for real serving; this is a diagnostic.
ep_args=(--enable-expert-parallel)
if [[ "${EP:-1}" == "0" ]]; then
    ep_args=()
    echo "expert parallelism disabled (EP=0)." >&2
fi

# PREFILL_MC2=1 sizes the MC2 buffers from max-num-batched-tokens instead of
# the largest captured graph size (set_mc2_tokens_capacity is its only reader).
# That size is where select_moe_comm_method stops using MC2 and falls to
# all-to-all, and that switch-over is where long prompts start answering with
# an immediate EOS -- so moving it is how the causal link gets tested. Here it
# goes from 400 to 4096, clamped by the 512-tokens-per-rank MC2 limit.
# That size also decides how much HCCL window MoeDistributeDispatch demands, and
# 2048MB is not enough for it: the tiling check asks for 4433MB and fails with
# EZ1008 during profile_run, naming HCCL_BUFFSIZE_EP. So raise that alongside,
# e.g. HCCL_BUFFSIZE_EP=5120, and expect the window to come out of KV cache.
additional_config='{"enable_cpu_binding":true'
if [[ "${PREFILL_MC2:-0}" == "1" ]]; then
    additional_config+=',"enable_prefill_mc2":true'
    echo "prefill MC2 on: MC2 capacity sized from max-num-batched-tokens." >&2
fi
additional_config+='}'

MODEL_PATH="${MODEL_PATH:-/mnt/share/weight/qwen3.5-397b-w4a4_multi}"
VLLM_PORT="${VLLM_PORT:-6969}"
MODEL_NAME="${MODEL_NAME:-qwen3.8}"
# 0.85 was needed while QFA quantized the whole KV cache on every step, which
# cost the aclgraph pool about 1.28x one attention layer's bf16 cache in
# transient tensors. On junlin-qfa that quantization is gone -- the cache is
# stored as MXFP8 -- so 0.95 may well fit again. Nobody has measured it since;
# if capture reports no available memory, drop to 0.85 and say so.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"

# QFA=1 runs the causal full-attention path on the vendored QuantFlashAttn
# instead of FIA (VLLM_ASCEND_ENABLE_QFA). It only takes effect on a C8-MXFP KV
# cache -- the checkpoint's quant description has to say
# kv_cache_type=K_DYNAMIC_V_STATIC_MXFP8_PER_CHANNEL and carry a per-layer
# v_proj.kv_cache_scale -- because that cache is the only thing QFA can read.
# With a bf16 cache the flag is ignored and the run is the FIA baseline.
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
    # VLLM_ASCEND_DISABLE_C8_MXFP only exists on branches carrying the C8 kill
    # switch (junlin-qfa-c8switch). Everywhere else exporting it is a silent
    # no-op: the server comes up with the C8 cache still on, and FIA then dies
    # at head_dim 256 with an EZ0010 that names neither C8 nor this switch. One
    # run was already lost reading that error as a result rather than a
    # misconfiguration, so check the installed package instead of hoping.
    # find_spec locates vllm_ascend without executing its __init__.
    envs_py="$(python3 -c 'import importlib.util, pathlib; s = importlib.util.find_spec("vllm_ascend"); print(pathlib.Path(s.origin).parent / "envs.py")' 2>/dev/null || true)"
    if [[ -n "$envs_py" && -r "$envs_py" ]] && ! grep -q VLLM_ASCEND_DISABLE_C8_MXFP "$envs_py"; then
        echo "ERROR: C8=0 needs VLLM_ASCEND_DISABLE_C8_MXFP, which the installed vllm-ascend does not define." >&2
        echo "       ($envs_py) Serve from the junlin-qfa-c8switch worktree, or drop C8=0." >&2
        exit 2
    fi
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
# Lowering it is the only way to move that constant without touching code, and
# the graph-capture crash reproduces nowhere else, so single-variable
# experiments have to run here rather than in a standalone script.
# MAX_NUM_SEQS bounds the batch, and with it the plan's per-core split and the
# graph plan below.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-133120}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-100}"
if ! [[ "$MAX_NUM_SEQS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_NUM_SEQS must be a positive integer, got '$MAX_NUM_SEQS'." >&2
    exit 1
fi

# MTP is num_speculative_tokens, not a flag: MTP=3 proposes three draft tokens
# per step, MTP=0 turns speculative decoding off. It sets the width of a decode
# step -- one accepted token plus MTP drafts -- which is what the graph plan
# below is built on.
MTP="${MTP:-3}"
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
# multiple, which is why the old hand-written list's leading 1 only ever
# merged into 4. The count is MAX_NUM_SEQS graphs whatever MTP is, so capture
# time and pool memory scale with the batch bound; CAPTURE_SIZES still
# overrides the whole list for single-variable experiments, as does
# CUDAGRAPH_MODE for keeping attention out of the graph entirely.
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
    --data-parallel-size 1 \
    --tensor-parallel-size 8 \
    "${ep_args[@]}" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 16384 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs "{\"enable_thinking\": $enable_thinking}" \
    --compilation-config "$compilation_config" \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./profiling", "torch_profiler_with_stack": false}' \
    "${spec_args[@]}" \
    --trust-remote-code \
    --async-scheduling \
    --allowed-local-media-path / \
    --mm-processor-cache-gb 0 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --additional-config "$additional_config" \
    "${qfa_args[@]}"
