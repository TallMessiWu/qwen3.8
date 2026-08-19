#!/usr/bin/env bash
# Install the host-mounted vLLM-Ascend checkout inside a newly created
# container. The /home bind mount must already be active.

set -euo pipefail

repo=/home/hajimi/qwen3.8/vllm-ascend/main

if [[ ! -d "$repo/csrc" || ! -f "$repo/pyproject.toml" ]]; then
    echo "ERROR: expected vLLM-Ascend checkout not found at $repo" >&2
    exit 2
fi

if [[ -f /home/hajimi/proxy.sh ]]; then
    source /home/hajimi/proxy.sh
fi

echo "Removing stale vLLM-Ascend C++ build outputs..."
rm -rf -- "$repo/csrc/output" "$repo/csrc/build_out"

cd "$repo"
python3 -m pip install -v --no-build-isolation -e . \
    --index-url https://mirrors.aliyun.com/pypi/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --trusted-host mirrors.aliyun.com \
    --trusted-host download.pytorch.org \
    --trusted-host download-r2.pytorch.org

python3 -c 'import importlib.metadata as m, importlib.util as u; print("vllm=" + m.version("vllm")); print("vllm-ascend=" + m.version("vllm-ascend")); print("vllm_ascend=" + str(u.find_spec("vllm_ascend").origin))'
