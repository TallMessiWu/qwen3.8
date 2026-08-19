#!/usr/bin/env bash
# Create the privileged A5 serving container from the image built by the root
# Dockerfile. Run this script on the host, not inside a container.

set -euo pipefail

IMAGE="${IMAGE:-qwen3.8-vllm-ascend:a5-20260819}"
CONTAINER_NAME="${CONTAINER_NAME:-hajimi-vllm}"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is not installed or not in PATH." >&2
    exit 2
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: image '$IMAGE' does not exist; build the root Dockerfile first." >&2
    exit 2
fi
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    echo "ERROR: container '$CONTAINER_NAME' already exists; refusing to replace it." >&2
    exit 2
fi

exec docker run --name "$CONTAINER_NAME" \
    --runtime=runc \
    --user root \
    --interactive \
    --tty \
    --detach \
    --net=host \
    --pid=host \
    --privileged=true \
    --shm-size=2g \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/ummu \
    --device=/dev/uburma \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --volume=/usr/local/Ascend/driver:/usr/local/Ascend/driver \
    --volume=/usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    --volume=/root/host:/root/host \
    --volume=/usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    --volume=/usr/local/sbin:/usr/local/sbin \
    --volume=/usr/local/dcmi:/usr/local/dcmi \
    --volume=/var/log/npu:/usr/slog \
    --volume=/mnt:/mnt \
    --volume=/data:/data \
    --volume=/etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
    --volume=/home:/home \
    --volume=/etc/hixlep:/etc/hixlep \
    --volume=/usr/lib64:/usr/lib64 \
    "$IMAGE" bash
