#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NODE_RANK="${NODE_RANK:-3}"
export LOCAL_IP="${LOCAL_IP:-141.61.52.191}"
export NODE0_IP="${NODE0_IP:-141.61.52.179}"
export NIC_NAME="${NIC_NAME:-enp35s0f2}"

exec bash "$script_dir/serve_qwen3.8_2.4t_4node.sh"
