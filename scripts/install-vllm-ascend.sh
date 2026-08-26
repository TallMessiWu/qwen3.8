#!/usr/bin/env bash
# Install vLLM-Ascend inside a newly created container, either from a mounted
# checkout or from a requested package version.

set -euo pipefail

repo=/home/hajimi/qwen3.8/vllm-ascend/feat-qfa-mxfp8-attn
proxy_file=/home/hajimi/proxy.sh
python_bin=python3
vllm_version=0.27.1
vllm_ascend_version=
pip_index_url=https://mirrors.aliyun.com/pypi/simple
pytorch_index_url=https://download.pytorch.org/whl/cpu

usage() {
    printf '%s\n' \
        "Usage: bash $0 [OPTIONS]" \
        "" \
        "Options:" \
        "  --vllm-ascend-repo PATH          Checkout to install in editable mode" \
        "  --proxy-file PATH                Optional proxy script" \
        "  --python-bin PATH                Python executable" \
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
        --vllm-ascend-repo | --proxy-file | --python-bin | --vllm-version | \
            --vllm-ascend-version | --pip-index-url | --pytorch-index-url) ;;
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
        --vllm-ascend-repo) repo="$value" ;;
        --proxy-file) proxy_file="$value" ;;
        --python-bin) python_bin="$value" ;;
        --vllm-version) vllm_version="$value" ;;
        --vllm-ascend-version) vllm_ascend_version="$value" ;;
        --pip-index-url) pip_index_url="$value" ;;
        --pytorch-index-url) pytorch_index_url="$value" ;;
    esac
done

if [[ -z "$vllm_ascend_version" && (! -d "$repo/csrc" || ! -f "$repo/pyproject.toml") ]]; then
    echo "ERROR: expected vLLM-Ascend checkout not found at $repo" >&2
    exit 2
fi

if [[ -f "$proxy_file" ]]; then
    source "$proxy_file"
fi

pip_source_args=(
    --index-url "$pip_index_url"
    --extra-index-url "$pytorch_index_url"
    --trusted-host mirrors.aliyun.com
    --trusted-host download.pytorch.org
    --trusted-host download-r2.pytorch.org
)

if [[ -n "$vllm_version" ]]; then
    "$python_bin" -m pip install --force-reinstall "vllm==$vllm_version" \
        --no-deps "${pip_source_args[@]}"
fi

if [[ -n "$vllm_ascend_version" ]]; then
    "$python_bin" -m pip install -v --no-build-isolation \
        "vllm-ascend==$vllm_ascend_version" "${pip_source_args[@]}"
else
    echo "Synchronizing vLLM-Ascend submodules..."
    git -C "$repo" submodule sync --recursive
    git -C "$repo" submodule update --init --recursive
    echo "Removing stale vLLM-Ascend C++ build outputs..."
    rm -rf -- "$repo/csrc/output" "$repo/csrc/build_out"
    cd "$repo"
    "$python_bin" -m pip install -v --no-build-isolation -e . \
        "${pip_source_args[@]}"
fi

"$python_bin" -c 'import importlib.metadata as m, importlib.util as u; print("vllm=" + m.version("vllm")); print("vllm-ascend=" + m.version("vllm-ascend")); print("vllm_ascend=" + str(u.find_spec("vllm_ascend").origin))'
