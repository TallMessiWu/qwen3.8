#!/bin/bash
# Qwen3.8-2.4T-A95B / 8 × Atlas 800 A3 (64GB × 16) / BF16 在线服务启动脚本
#
# 并行策略：TP16 (机内) × DP8 (跨机) + EP128，attention 走 TP16+DP8，MoE 走全局 EP128。
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

# 显存预算（单 die 64GB，共 128 die = 8TB）：
#   BF16 权重 2.4T 参数 ≈ 4.8TB，摊到 128 die ≈ 37.5GB/die
#   64GB × 0.9 = 57.6GB 可用 → 扣掉权重后剩 ≈ 20GB/die 给 KV cache + 激活 + 通信 buffer
#   起服务失败(OOM)就先降 MAX_MODEL_LEN 或 MAX_NUM_SEQS，再考虑降 GPU_MEM_UTIL
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
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

echo "[INFO] 权重架构：$(grep -o '"architectures"[^]]*]' "$MODEL_PATH/config.json" | head -1)"

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
