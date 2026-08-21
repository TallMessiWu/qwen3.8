#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
safe_repr_dir="${script_dir}/safe_tensor_repr"

export QWEN38_SAFE_TENSOR_REPR=1
export PYTHONPATH="${safe_repr_dir}${PYTHONPATH:+:${PYTHONPATH}}"

echo "SAFE_TENSOR_REPR=enabled"
exec "$@"
