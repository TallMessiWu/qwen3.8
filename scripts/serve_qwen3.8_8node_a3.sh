#!/bin/bash
# Qwen3.8-2.4T-A95B / 8 × Atlas 800 A3 (64GB × 16) / BF16 在线服务启动脚本
#
# 模型架构（取自官方 config.json，architectures = Qwen3_5MoeForCausalLM）：
#   92 层混合骨干 = 23 × [3 × linear_attention(Gated DeltaNet) + 1 × full_attention(GQA)]
#   hidden 8192 / vocab 248320 / max_position_embeddings 262144
#   GQA:  64 Q heads, 4 KV heads, head_dim 256, partial_rotary_factor 0.25
#   GDN:  linear_num_key_heads 16, linear_num_value_heads 128, head_dim 128, conv_kernel 4
#   MoE:  512 experts, top-10 + 1 shared, moe_intermediate_size 2048
#   MTP:  mtp_num_hidden_layers 1（投机解码可用，见下方 ENABLE_MTP）
#
# 并行策略：TP16 (机内) × DP8 (跨机) + EP128，attention 走 TP16+DP8，MoE 走全局 EP128。
# TP16 整除性已核对：64/16=4, 128/16=8, 16/16=1, 512/128=4 全部整除；
# 4 个 KV heads 少于 TP16，vLLM 会把 KV head 复制成 16 份（16%4=0 合法），KV cache 有 4× 冗余。
#
# 用法：8 台机器上跑同一个脚本，只改 NODE_RANK。node0 是 master，对外提供 API。
#
#   # node0（master，提供 8000 端口的 API）
#   NODE_RANK=0 LOCAL_IP=10.0.0.10 NIC_NAME=enp67s0f0 MASTER_IP=10.0.0.10 bash serve_qwen3.8_8node_a3.sh
#   # node1..node7（headless，不起 API server）
#   NODE_RANK=1 LOCAL_IP=10.0.0.11 NIC_NAME=enp67s0f0 MASTER_IP=10.0.0.10 bash serve_qwen3.8_8node_a3.sh
#   ...
#   NODE_RANK=7 LOCAL_IP=10.0.0.17 NIC_NAME=enp67s0f0 MASTER_IP=10.0.0.10 bash serve_qwen3.8_8node_a3.sh
#
# 先起 node0，等它打印出监听 DP RPC 端口后再起 node1-7（顺序无所谓，但 master 必须先起）。
# LOCAL_IP / NIC_NAME 用 `ifconfig` 或 `ip addr` 查，NIC_NAME 必须是 LOCAL_IP 对应的那张网卡。

set -euo pipefail

# ============================ 可调参数 ============================
MODEL_PATH="${MODEL_PATH:-/home/weights/Qwen3.8-2.4T-A95B}"
SERVED_NAME="${SERVED_NAME:-qwen3.8}"
API_PORT="${API_PORT:-8000}"
DP_RPC_PORT="${DP_RPC_PORT:-13389}"

# 显存预算（单 die 64GB，共 128 die = 8TB），按 config.json 实算：
#   MoE 权重  2.37T 参数走 EP128，每 die 18.5B × 2B ≈ 37.1GB
#   非 MoE    48.4B 参数（embed/lm_head/GQA/GDN/shared expert）在每个 DP rank 内按 TP16 切，
#             每 die ≈ 6.1GB —— 注意 DP8 会让这部分在 8 个 rank 上各存一份
#   合计权重 ≈ 43.2GB/die；64GB × 0.9 = 57.6GB → 只剩 ≈ 14.4GB/die 给 KV cache + GDN state + 激活
#
#   两类缓存的吃法完全不同，调参时分清楚：
#   - KV cache 只有 23 层 full attention 有，每 die 1 个（复制后的）KV head
#     ≈ 23KB/token，随「上下文长度 × 并发」增长
#   - GDN state 是 69 层的常数状态，≈ 37MB/序列/die，**只随 MAX_NUM_SEQS 增长，与序列长度无关**，
#     且按 max_num_seqs 预分配。所以调大并发比调长上下文更容易 OOM
#   起服务 OOM 就先降 MAX_NUM_SEQS，再降 MAX_MODEL_LEN，最后才动 GPU_MEM_UTIL
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

# MTP 投机解码（config.json 里 mtp_num_hidden_layers=1，vLLM 侧 Qwen3_5MoeMTP 已注册）。
# 默认关：BF16 权重已占掉 3/4 显存，投机会同时增加 MTP 层权重和 GDN state 的 num_spec 维度。
# 先把基础服务跑通、确认余量后再置 1 打开。
ENABLE_MTP="${ENABLE_MTP:-0}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
# =================================================================

for v in NODE_RANK LOCAL_IP NIC_NAME MASTER_IP; do
    if [ -z "${!v:-}" ]; then
        echo "[ERROR] 环境变量 $v 未设置。用法见脚本头部注释。" >&2
        exit 1
    fi
done

if [ "$NODE_RANK" -lt 0 ] || [ "$NODE_RANK" -gt 7 ]; then
    echo "[ERROR] NODE_RANK 必须在 0..7 之间，当前为 $NODE_RANK" >&2
    exit 1
fi
if [ "$NODE_RANK" -eq 0 ] && [ "$LOCAL_IP" != "$MASTER_IP" ]; then
    echo "[ERROR] node0 是 master，LOCAL_IP($LOCAL_IP) 必须等于 MASTER_IP($MASTER_IP)" >&2
    exit 1
fi

# ---- 前置检查：权重可读 / 卡数够 / 网卡对得上 ----
if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] 权重目录不存在：$MODEL_PATH（8 台机器都要能读到，共享盘或各机本地副本均可）" >&2
    exit 1
fi
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "[ERROR] $MODEL_PATH/config.json 不存在，权重目录不完整" >&2
    exit 1
fi

npu_count=$(npu-smi info -l 2>/dev/null | grep -c "NPU ID" || echo 0)
echo "[INFO] node$NODE_RANK 检测到 NPU 芯片数：$npu_count（期望 16）"
if [ "$npu_count" -ne 16 ]; then
    echo "[WARN] 芯片数不是 16，TP16 会起不来。确认是 A3 (64GB × 16) 机型再继续。" >&2
fi

if ! ip addr show "$NIC_NAME" 2>/dev/null | grep -q "$LOCAL_IP"; then
    echo "[ERROR] 网卡 $NIC_NAME 上没有找到 IP $LOCAL_IP，检查 NIC_NAME/LOCAL_IP 是否配对" >&2
    exit 1
fi

# 架构必须是 Qwen3_5MoeForCausalLM：vLLM registry 和 vllm-ascend 的 patch_qwen3_5 都按这个名字挂钩
if ! grep -q "Qwen3_5MoeForCausalLM" "$MODEL_PATH/config.json"; then
    echo "[WARN] config.json 里没有 Qwen3_5MoeForCausalLM，实际为：" >&2
    grep -o '"architectures"[^]]*]' "$MODEL_PATH/config.json" | head -1 >&2
    echo "[WARN] 当前 vLLM 可能不支持该架构，启动会在模型加载阶段失败。" >&2
else
    echo "[INFO] 架构校验通过：Qwen3_5MoeForCausalLM"
fi

# ---- 通信配置（A3 口径）----
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export HCCL_BUFFSIZE=2048
export HCCL_OP_EXPANSION_MODE="AIV"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# ---- 超时：4.8TB BF16 权重加载 + 128 卡 HCCL 建链都很慢，超时必须放大 ----
export HCCL_CONNECT_TIMEOUT=7200
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_RPC_TIMEOUT=1800000

# ---- 内存 / 线程 ----
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export TASK_QUEUE_ENABLE=1
export ACL_OP_INIT_MODE=1
# jemalloc 降低 host 侧内存碎片，装了才 preload
if [ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]; then
    export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
fi

# ---- vllm-ascend 特性开关 ----
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

# master 起 API server，其余节点 headless
extra_args=()
if [ "$NODE_RANK" -ne 0 ]; then
    extra_args+=(--headless)
else
    extra_args+=(--api-server-count 4)
fi

if [ "$ENABLE_MTP" = "1" ]; then
    echo "[INFO] 启用 MTP 投机解码，num_speculative_tokens=$NUM_SPEC_TOKENS"
    extra_args+=(--speculative-config \
        "{\"method\": \"qwen3_5_mtp\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"enforce_eager\": true}")
fi

echo "[INFO] 启动 node$NODE_RANK：TP16 × DP8 (start-rank=$NODE_RANK)，master=$MASTER_IP:$DP_RPC_PORT"

vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --served-model-name "$SERVED_NAME" \
    --trust-remote-code \
    --tensor-parallel-size 16 \
    --data-parallel-size 8 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank "$NODE_RANK" \
    --data-parallel-address "$MASTER_IP" \
    --data-parallel-rpc-port "$DP_RPC_PORT" \
    --enable-expert-parallel \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --block-size 128 \
    --no-enable-prefix-caching \
    --seed 1024 \
    --safetensors-load-strategy prefetch \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 64}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true}' \
    "${extra_args[@]}"
