#!/usr/bin/env python3
"""Decide whether a given node/rank topology can actually hold this checkpoint.

Context: the four-node 2.4T launch dies during KV cache sizing with

  Available KV cache memory: -4.90 GiB
  ValueError: No available memory for the cache blocks.

vLLM-Ascend computes that number as

  available_kv = total_hbm * gpu_memory_utilization
               - (weights + peak_activation + non_torch)      # worker.py:600

so a negative value means the *weights alone plus their working set* already
exceed the requested budget.  Answering "do four of these machines have enough
memory" therefore needs three separate numbers, not one:

  1. per-rank weight bytes      -- decided by the checkpoint and the topology
  2. per-rank non-weight bytes  -- activation peak, HCCL buffers, ACL context
  3. per-rank KV bytes needed   -- decided by max_model_len x max_num_seqs

This script computes (1) exactly from the safetensors headers, computes (3)
exactly from config.json (hybrid model: full-attention layers hold a KV cache,
linear-attention layers hold a fixed per-sequence GDN state), and lets you feed
(2) in from a real run so the verdict is not built on a guess.

Reads only safetensors headers (8-byte length prefix + JSON, no tensor
payload), never imports torch/vllm, never touches the NPU.

Usage:
  python3 plan_hbm_budget.py [MODEL_PATH] [--tp 8] [--dp 4] [--hbm-gib 96]
                             [--util 0.85] [--max-model-len 131072]
                             [--max-num-seqs 8] [--sample-shards N]
                             [--non-weight-gib X | --observed-kv-gib Y]
                             [--compare-topologies]

With no measured input it derives the non-weight footprint from
--observed-kv-gib (the "Available KV cache memory" line of a failed run), which
is the most accurate source available before the engine ever starts.

Final line is [GREEN] (exit 0) or [RED] (exit 1).
"""

import argparse
import json
import struct
import sys
from pathlib import Path

DEFAULT_MODEL_PATH = "/mnt/share/weight/Qwen3.8-2.4T-A95B-mxfp8"

GIB = float(1 << 30)

# safetensors dtype -> bytes per element.
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1, "F8_E8M0": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


def read_safetensors_header(path):
    """Return the safetensors JSON header without reading any tensor payload."""
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(header_len))


def load_config(model_dir):
    """Return the text config, unwrapping the nested multimodal layout."""
    config = json.loads((Path(model_dir) / "config.json").read_text())
    text = config.get("text_config") or config
    return config, text


def is_expert_tensor(name):
    """Expert weights are sharded over EP; everything else is replicated per DP group."""
    return ".experts." in name and "shared_expert" not in name


def scan_checkpoint(model_dir, sample_shards):
    """Classify every tensor into expert / dense-shardable / replicated bytes."""
    shards = sorted(Path(model_dir).glob("*.safetensors"))
    if not shards:
        raise SystemExit("[RED] no *.safetensors under " + str(model_dir))

    scanned = shards if sample_shards <= 0 else shards[:sample_shards]
    totals = {"expert": 0, "dense": 0, "replicated": 0}
    unknown_dtypes = set()

    for index, path in enumerate(scanned, 1):
        for name, entry in read_safetensors_header(path).items():
            if name == "__metadata__":
                continue
            dtype = entry["dtype"]
            if dtype not in DTYPE_BYTES:
                unknown_dtypes.add(dtype)
                continue
            shape = entry["shape"]
            nbytes = DTYPE_BYTES[dtype]
            for dim in shape:
                nbytes *= dim
            if is_expert_tensor(name):
                totals["expert"] += nbytes
            elif len(shape) >= 2:
                totals["dense"] += nbytes
            else:
                # 1-D norms/biases/A_log/dt_bias are not TP-split; every rank keeps a copy.
                totals["replicated"] += nbytes
        if index % 25 == 0 or index == len(scanned):
            print("  scanned %d/%d shards" % (index, len(scanned)))

    scale = 1.0
    if sample_shards > 0 and len(scanned) < len(shards):
        scanned_bytes = sum(p.stat().st_size for p in scanned)
        all_bytes = sum(p.stat().st_size for p in shards)
        scale = all_bytes / float(scanned_bytes)
        print("  extrapolating %d -> %d shards (x%.3f)" % (len(scanned), len(shards), scale))
        totals = {k: v * scale for k, v in totals.items()}

    if unknown_dtypes:
        print("  WARNING: skipped unknown dtypes " + ", ".join(sorted(unknown_dtypes)))
    return totals, len(shards), sum(p.stat().st_size for p in shards)


def per_rank_weight_bytes(totals, tp, ep):
    """Expert bytes split over EP, 2-D dense over TP, 1-D replicated everywhere.

    Note that DP does NOT reduce the dense part: each DP group holds a full
    TP-sharded copy of attention / GDN / embedding weights.
    """
    return totals["expert"] / ep + totals["dense"] / tp + totals["replicated"]


def kv_bytes_per_token(text, tp, kv_dtype_bytes):
    """Per-rank KV cache bytes for one token, summed over full-attention layers."""
    layer_types = text.get("layer_types")
    num_layers = text["num_hidden_layers"]
    if layer_types:
        full_layers = sum(1 for t in layer_types if t == "full_attention")
    else:
        full_layers = num_layers
    kv_heads = text.get("num_key_value_heads", text["num_attention_heads"])
    head_dim = text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"]
    # vLLM replicates KV heads when they cannot cover TP; a rank never holds < 1 head.
    heads_per_rank = max(1.0, kv_heads / float(tp))
    return full_layers, int(full_layers * 2 * heads_per_rank * head_dim * kv_dtype_bytes)


def gdn_bytes_per_seq(text, tp, mamba_dtype_bytes, num_spec):
    """Per-rank GDN state bytes for one sequence (constant in sequence length).

    Mirrors MambaStateShapeCalculator.gated_delta_net_state_shape.
    """
    layer_types = text.get("layer_types")
    if not layer_types:
        return 0, 0
    linear_layers = sum(1 for t in layer_types if t == "linear_attention")
    if not linear_layers:
        return 0, 0
    num_k = text["linear_num_key_heads"]
    num_v = text["linear_num_value_heads"]
    dim_k = text["linear_key_head_dim"]
    dim_v = text["linear_value_head_dim"]
    kernel = text["linear_conv_kernel_dim"]

    conv_dim = dim_k * num_k * 2 + dim_v * num_v
    conv_state = (conv_dim / float(tp)) * (kernel - 1 + num_spec)
    temporal_state = (num_v / float(tp)) * dim_v * dim_k
    return linear_layers, int(linear_layers * (conv_state + temporal_state) * mamba_dtype_bytes)


def gib(value):
    return value / GIB


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_path", nargs="?", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tp", type=int, default=8, help="tensor-parallel-size (per DP group)")
    parser.add_argument("--dp", type=int, default=4, help="data-parallel-size; EP = tp * dp")
    parser.add_argument("--hbm-gib", type=float, default=96.0, help="per-device HBM as torch sees it")
    parser.add_argument("--util", type=float, default=0.85, help="--gpu-memory-utilization")
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--kv-dtype-bytes", type=int, default=2, help="2 for bf16/fp16 KV cache")
    parser.add_argument("--mamba-dtype-bytes", type=int, default=2)
    parser.add_argument("--num-spec", type=int, default=0, help="num_speculative_tokens when MTP is on")
    parser.add_argument("--sample-shards", type=int, default=0, help="scan only N shards and extrapolate")
    parser.add_argument("--non-weight-gib", type=float, default=None,
                        help="measured peak activation + non-torch per rank")
    parser.add_argument("--observed-kv-gib", type=float, default=None,
                        help="the 'Available KV cache memory' value of a real run, to back out non-weight bytes")
    parser.add_argument("--compare-topologies", action="store_true",
                        help="also print TP16/DP2 and TP32/DP1 for the same 32 ranks")
    args = parser.parse_args()

    ep = args.tp * args.dp
    config, text = load_config(args.model_path)
    print("=== checkpoint ===")
    print("  path            = " + str(args.model_path))
    print("  model_type      = " + str(config.get("model_type")) + " / " + str(text.get("model_type")))
    print("  num_hidden_layers = %s   num_experts = %s" % (text.get("num_hidden_layers"), text.get("num_experts")))
    print("  hidden_size = %s  heads = %s  kv_heads = %s  head_dim = %s"
          % (text.get("hidden_size"), text.get("num_attention_heads"),
             text.get("num_key_value_heads"), text.get("head_dim")))

    print("=== scanning safetensors headers ===")
    totals, num_shards, disk_bytes = scan_checkpoint(args.model_path, args.sample_shards)
    counted = totals["expert"] + totals["dense"] + totals["replicated"]
    print("  shards on disk  = %d (%.1f GiB)" % (num_shards, gib(disk_bytes)))
    print("  expert weights  = %8.1f GiB  (split over EP=%d)" % (gib(totals["expert"]), ep))
    print("  dense 2-D       = %8.1f GiB  (split over TP=%d, replicated across DP=%d)"
          % (gib(totals["dense"]), args.tp, args.dp))
    print("  replicated 1-D  = %8.3f GiB  (every rank keeps a full copy)" % gib(totals["replicated"]))
    print("  counted total   = %8.1f GiB  (%.1f%% of on-disk bytes)"
          % (gib(counted), 100.0 * counted / disk_bytes if disk_bytes else 0.0))

    weight_bytes = per_rank_weight_bytes(totals, args.tp, ep)

    print("=== per-rank memory budget (TP=%d DP=%d EP=%d) ===" % (args.tp, args.dp, ep))
    requested = args.hbm_gib * GIB * args.util
    print("  HBM per device        = %7.2f GiB" % args.hbm_gib)
    print("  requested (util=%.2f)  = %7.2f GiB" % (args.util, gib(requested)))
    print("  weights               = %7.2f GiB" % gib(weight_bytes))

    if args.non_weight_gib is not None:
        non_weight = args.non_weight_gib * GIB
        source = "--non-weight-gib"
    elif args.observed_kv_gib is not None:
        # available_kv = requested - (weights + non_weight)  =>  solve for non_weight
        non_weight = requested - args.observed_kv_gib * GIB - weight_bytes
        source = "back-computed from --observed-kv-gib"
    else:
        non_weight = None
        source = None

    if non_weight is None:
        print("  activation+non-torch  =       ?  GiB  (pass --observed-kv-gib or --non-weight-gib)")
        print()
        print("  Re-run with the 'Available KV cache memory' number from a real launch, e.g.")
        print("    --observed-kv-gib -4.90")
        print("[RED] no measured non-weight footprint; verdict undecided")
        return 1

    print("  activation+non-torch  = %7.2f GiB  (%s)" % (gib(non_weight), source))
    if args.non_weight_gib is None:
        print("      NOTE: this residual also absorbs any error in the weight estimate above.")
        print("      Cross-check it against the 'Actual usage: ... for weights, ... for peak")
        print("      activation, ... for non-torch memory' line a successful launch prints.")
    non_kv = weight_bytes + non_weight
    available_kv = requested - non_kv
    headroom_full = args.hbm_gib * GIB - non_kv
    print("  non-KV total          = %7.2f GiB" % gib(non_kv))
    print("  available KV          = %7.2f GiB   <-- what vLLM prints" % gib(available_kv))
    print("  KV at util=1.00       = %7.2f GiB   <-- hard physical ceiling" % gib(headroom_full))

    full_layers, kv_per_token = kv_bytes_per_token(text, args.tp, args.kv_dtype_bytes)
    linear_layers, gdn_per_seq = gdn_bytes_per_seq(text, args.tp, args.mamba_dtype_bytes, args.num_spec)

    print("=== per-rank KV demand ===")
    print("  full_attention layers = %d, linear_attention layers = %d" % (full_layers, linear_layers))
    print("  KV cache per token    = %.2f KiB" % (kv_per_token / 1024.0))
    print("  GDN state per seq     = %.2f MiB" % (gdn_per_seq / (1024.0 * 1024.0)))

    tokens_wanted = args.max_num_seqs * args.max_model_len
    needed = tokens_wanted * kv_per_token + args.max_num_seqs * gdn_per_seq
    minimum = args.max_model_len * kv_per_token + gdn_per_seq  # one full-length sequence
    print("  need for %d seq x %d tok = %.2f GiB" % (args.max_num_seqs, args.max_model_len, gib(needed)))
    print("  need for 1 seq x %d tok  = %.2f GiB  <-- engine refuses to start below this"
          % (args.max_model_len, gib(minimum)))

    print("=== verdict ===")
    if available_kv > 0:
        usable_tokens = max(0.0, (available_kv - args.max_num_seqs * gdn_per_seq) / kv_per_token)
        print("  at util=%.2f: %.0f KV tokens per rank (%.1f full-length seqs)"
              % (args.util, usable_tokens, usable_tokens / args.max_model_len))
    ceiling_tokens = max(0.0, (headroom_full - args.max_num_seqs * gdn_per_seq) / kv_per_token)
    print("  physical ceiling: %.0f KV tokens per rank (%.1f full-length seqs)"
          % (ceiling_tokens, ceiling_tokens / args.max_model_len))
    if headroom_full > 0:
        util_needed = (non_kv + minimum) / (args.hbm_gib * GIB)
        print("  minimum --gpu-memory-utilization to start = %.4f" % util_needed)

    if args.compare_topologies:
        print("=== topology comparison (same 32 ranks) ===")
        print("  %-12s %10s %10s %14s" % ("layout", "weights", "KV@1.00", "full-len seqs"))
        for tp, dp in ((8, 4), (16, 2), (32, 1)):
            w = per_rank_weight_bytes(totals, tp, tp * dp)
            head = args.hbm_gib * GIB - w - non_weight
            _, kv_tok = kv_bytes_per_token(text, tp, args.kv_dtype_bytes)
            _, gdn = gdn_bytes_per_seq(text, tp, args.mamba_dtype_bytes, args.num_spec)
            seqs = (head - args.max_num_seqs * gdn) / (args.max_model_len * kv_tok) if head > 0 else 0.0
            print("  TP%-2d DP%-2d    %7.2f G %8.2f G %13.1f" % (tp, dp, gib(w), gib(head), max(0.0, seqs)))
        print("  (non-weight footprint held constant; TP16/TP32 need cross-node TP links)")

    if headroom_full < minimum:
        print("[RED] this topology cannot hold one full-length sequence; add nodes, "
              "shrink max-model-len, or shard the dense part harder")
        return 1
    if available_kv < minimum:
        print("[RED] weights fit but util=%.2f leaves too little; raise --gpu-memory-utilization "
              "or cut the non-weight footprint" % args.util)
        return 1
    print("[GREEN] topology holds the weights and at least one full-length sequence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
