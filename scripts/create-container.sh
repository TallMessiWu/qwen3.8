#!/usr/bin/env bash
# Create the privileged A5 serving container from the vendor image, then
# install the host-mounted vLLM-Ascend checkout. Run this script on the host,
# not inside a container.

set -euo pipefail

IMAGE="${IMAGE:-vllm-ascend:dev-26.1.0.day20260817-A5-py311-openEuler24.03-lts-aarch64}"
CONTAINER_NAME="${CONTAINER_NAME:-hajimi-vllm}"
INSTALL_SCRIPT=/home/hajimi/qwen3.8/scripts/install-vllm-ascend.sh
VLLM_ASCEND_REPO=/home/hajimi/qwen3.8/vllm-ascend/main
PROXY_FILE=/home/hajimi/proxy.sh
SHELL_WORKDIR=/home/hajimi/qwen3.8/scripts
SHELL_FALLBACK_DIR=/home/hajimi
PYTHON_BIN=python3
VLLM_VERSION=0.27.1
VLLM_ASCEND_VERSION=
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

usage() {
    printf '%s\n' \
        "Usage: bash $0 [OPTIONS]" \
        "" \
        "Options:" \
        "  --image IMAGE                    Container image" \
        "  --container-name NAME            Container name" \
        "  --install-script PATH            Installer path visible inside the container" \
        "  --vllm-ascend-repo PATH          vLLM-Ascend checkout visible inside the container" \
        "  --proxy-file PATH                Optional proxy script visible inside the container" \
        "  --shell-workdir PATH             Initial directory for interactive root shells" \
        "  --shell-fallback-dir PATH        Directory used when shell workdir is absent" \
        "  --python-bin PATH                Python executable used by the installer" \
        "  --vllm-version VERSION           vLLM version to reinstall (default: 0.27.1)" \
        "  --vllm-ascend-version VERSION    Install this package version instead of the checkout" \
        "  --pip-index-url URL              Primary Python package index" \
        "  --pytorch-index-url URL          Extra PyTorch package index" \
        "  -h, --help                       Show this help"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
    esac

    option="${1%%=*}"
    case "$option" in
        --image | --container-name | --install-script | --vllm-ascend-repo | \
            --proxy-file | --shell-workdir | --shell-fallback-dir | --python-bin | \
            --vllm-version | --vllm-ascend-version | --pip-index-url | --pytorch-index-url) ;;
        *)
            echo "ERROR: unknown option '$option'." >&2
            usage >&2
            exit 2
            ;;
    esac

    if [[ "$1" == *=* ]]; then
        value="${1#*=}"
        shift
    elif [[ $# -ge 2 && "$2" != --* ]]; then
        value="$2"
        shift 2
    else
        value=
    fi
    if [[ -z "$value" ]]; then
        echo "ERROR: $option requires a non-empty value." >&2
        usage >&2
        exit 2
    fi

    case "$option" in
        --image) IMAGE="$value" ;;
        --container-name) CONTAINER_NAME="$value" ;;
        --install-script) INSTALL_SCRIPT="$value" ;;
        --vllm-ascend-repo) VLLM_ASCEND_REPO="$value" ;;
        --proxy-file) PROXY_FILE="$value" ;;
        --shell-workdir) SHELL_WORKDIR="$value" ;;
        --shell-fallback-dir) SHELL_FALLBACK_DIR="$value" ;;
        --python-bin) PYTHON_BIN="$value" ;;
        --vllm-version) VLLM_VERSION="$value" ;;
        --vllm-ascend-version) VLLM_ASCEND_VERSION="$value" ;;
        --pip-index-url) PIP_INDEX_URL="$value" ;;
        --pytorch-index-url) PYTORCH_INDEX_URL="$value" ;;
    esac
done

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is not installed or not in PATH." >&2
    exit 2
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: image '$IMAGE' does not exist; pull or load it first." >&2
    exit 2
fi
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    echo "ERROR: container '$CONTAINER_NAME' already exists; refusing to replace it." >&2
    exit 2
fi

docker run --name "$CONTAINER_NAME" \
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

if ! docker exec --user root "$CONTAINER_NAME" bash -c '
    {
        printf "if [[ -f %q ]]; then source %q; fi\n" "$1" "$1"
        printf "if [[ -d %q ]]; then cd %q; else cd %q; fi\n" "$2" "$2" "$3"
    } >> /root/.bashrc
' bash "$PROXY_FILE" "$SHELL_WORKDIR" "$SHELL_FALLBACK_DIR"; then
    echo "ERROR: failed to configure root's interactive shell in container '$CONTAINER_NAME'." >&2
    exit 1
fi

install_args=(
    bash "$INSTALL_SCRIPT"
    --vllm-ascend-repo "$VLLM_ASCEND_REPO"
    --proxy-file "$PROXY_FILE"
    --python-bin "$PYTHON_BIN"
    --pip-index-url "$PIP_INDEX_URL"
    --pytorch-index-url "$PYTORCH_INDEX_URL"
)
if [[ -n "$VLLM_VERSION" ]]; then
    install_args+=(--vllm-version "$VLLM_VERSION")
fi
if [[ -n "$VLLM_ASCEND_VERSION" ]]; then
    install_args+=(--vllm-ascend-version "$VLLM_ASCEND_VERSION")
fi

if ! docker exec --user root "$CONTAINER_NAME" "${install_args[@]}"; then
    echo "ERROR: vLLM-Ascend installation failed; container '$CONTAINER_NAME' was left running for inspection." >&2
    exit 1
fi

echo "Container '$CONTAINER_NAME' is running with vLLM-Ascend installed."
