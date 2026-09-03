#!/usr/bin/env python3
"""Decide whether N nodes of NPUs can hold a checkpoint, from safetensors headers.

Context: the 2.4T four-node launch dies with

  INFO worker.py:622 Available KV cache memory: -4.90 GiB
  ValueError: No available memory for the cache blocks.

vLLM computes (vllm_ascend/worker/worker.py:540-627, vllm/v1/worker/utils.py:517):

    requested    = HBM_total * gpu_memory_utilization
    non_kv_cache = weights + torch_peak_activation + non_torch(HCCL/workspace)
    available_kv = requested - non_kv_cache

A negative number means the weights plus the fixed overhead already exceed the
requested budget, so no amount of timeout tuning helps.  The question "do four
machines have enough HBM" therefore splits into three independent quantities:

  1. routed-expert weights   -> divided by EP size (= TP * DP), shrinks with nodes
  2. TP-shardable weights    -> divided by TP size, does NOT shrink with nodes
  3. replicated weights      -> held whole on every rank, never shrinks

Only (1) benefits from adding machines.  This script measures all three exactly
from the safetensors headers (8-byte length prefix + JSON, no tensor payload),
never imports torch/vllm, and never touches the NPU.

It can also back-solve the runtime overhead: pass the "Available KV cache
memory: X GiB" number from a failed run via --observed-kv-gib and the script
turns that single datum into a measured (activation + non-torch) figure, which
is the only term it cannot compute statically.  A failed run also prints
"Loading model weights took X GB" (model_runner_v1.py:3837, emitted before
profiling starts): pass it as --weights-gib and the residual stops absorbing
the on-disk-vs-runtime weight difference, so weights and overhead separate
cleanly without any further NPU time.  With that calibration it then
answers how many nodes, which max-model-len, and which utilization would work.

Usage:
  python3 estimate_hbm_budget.py [MODEL_PATH] \
      [--tp 8] [--dp 4] [--no-ep] [--nodes 4] [--npus-per-node 8] \
      [--hbm-gib 96] [--util 0.85] \
      [--max-model-len 131072] [--max-num-seqs 8] \
      [--kv-cache-dtype auto] \
      [--observed-kv-gib -4.90] [--observed-util 0.85] \
      [--weights-gib 74.03] \
      [--max-shards 0]

Final line is [GREEN] (exit 0) or [RED] (exit 1).
"""

import json
import re
import struct
import sys
from pathlib import Path

DEFAULT_MODEL_PATH = "/mnt/share/weight/Qwen3.8-2.4T-A95B-mxfp8"

GiB = float(1 << 30)

DTYPE_BITS = {
    "BOOL": 8, "U8": 8, "I8": 8, "F8_E4M3": 8, "F8_E5M2": 8, "F8_E8M0": 8,
    "I16": 16, "U16": 16, "F16": 16, "BF16": 16,
    "I32": 32, "U32": 32, "F32": 32,
    "I64": 64, "U64": 64, "F64": 64,
    "F4_E2M1": 4, "I4": 4, "U4": 4,
}

# A routed expert weight: ".experts.<id>." per-expert, or a fused 3-D
# ".experts.<name>" tensor.  Shared experts are NOT routed and stay TP-sharded.
ROUTED_EXPERT_RE = re.compile(r"\.experts\.")
SHARED_EXPERT_RE = re.compile(r"shared_expert")

# Names vLLM shards along a tensor-parallel axis.
TP_SHARDED_HINTS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj",
    "gate_proj", "up_proj", "down_proj", "gate_up_proj",
    "in_proj", "out_proj", "fc1", "fc2", "mlp.experts",  # in_proj* = GDN
    "embed_tokens", "lm_head", "attn.proj", "attn.qkv",
)
# Anything matching these stays whole on every rank.
REPLICATED_HINTS = (
    "norm", "layer_scale", "_bias", ".bias", "gate.weight", "mlp.gate.",
    "router", "e_score_correction", "A_log", "dt_bias", "conv1d",
)


def read_header(path):
    """Return the safetensors JSON header without reading any tensor payload."""
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(header_len))


def tensor_bytes(entry):
    """Exact on-disk bytes: prefer data_offsets, fall back to shape x dtype."""
    offsets = entry.get("data_offsets")
    if offsets and len(offsets) == 2:
        return int(offsets[1]) - int(offsets[0])
    bits = DTYPE_BITS.get(entry.get("dtype", ""))
    if bits is None:
        return 0
    numel = 1
    for dim in entry.get("shape", ()):
        numel *= int(dim)
    return numel * bits // 8


def classify(name):
    """Return 'expert' | 'tp' | 'replicated' for a checkpoint tensor name."""
    if ROUTED_EXPERT_RE.search(name) and not SHARED_EXPERT_RE.search(name):
        return "expert"
    for hint in REPLICATED_HINTS:
        if hint in name:
            return "replicated"
    for hint in TP_SHARDED_HINTS:
        if hint in name:
            return "tp"
    return "replicated"


def parse_args(argv):
    opts = {
        "model_path": DEFAULT_MODEL_PATH,
        "tp": 8, "dp": 4, "ep": True,
        "nodes": 4, "npus_per_node": 8,
        "hbm_gib": 96.0, "util": 0.85,
        "max_model_len": 131072, "max_num_seqs": 8,
        "kv_cache_dtype": "auto",
        "observed_kv_gib": None, "observed_util": None,
        "weights_gib": None,
        "max_shards": 0,
    }
    positional = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--no-ep":
            opts["ep"] = False
        elif arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            if key not in opts:
                print("unknown option: " + arg)
                raise SystemExit(2)
            i += 1
            if i >= len(argv):
                print("missing value for " + arg)
                raise SystemExit(2)
            value = argv[i]
            if key in ("model_path", "kv_cache_dtype"):
                opts[key] = value
            elif key in ("hbm_gib", "util", "observed_kv_gib", "observed_util", "weights_gib"):
                opts[key] = float(value)
            else:
                opts[key] = int(value)
        else:
            positional.append(arg)
        i += 1
    if positional:
        opts["model_path"] = positional[0]
    return opts


def collect_weight_bytes(model_dir, max_shards):
    """Sum on-disk bytes per sharding class over every safetensors shard."""
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        print("[RED] no *.safetensors under " + str(model_dir))
        return None

    limit = len(shards) if max_shards <= 0 else min(max_shards, len(shards))
    scale = len(shards) / limit
    print("  " + str(len(shards)) + " shard(s); reading " + str(limit) + " header(s)"
          + ("" if limit == len(shards) else "  (EXTRAPOLATED x%.2f)" % scale))

    totals = {"expert": 0, "tp": 0, "replicated": 0}
    counts = {"expert": 0, "tp": 0, "replicated": 0}
    samples = {"expert": None, "tp": None, "replicated": None}
    for idx, shard in enumerate(shards[:limit], start=1):
        for name, entry in read_header(shard).items():
            if name == "__metadata__":
                continue
            kind = classify(name)
            totals[kind] += tensor_bytes(entry)
            counts[kind] += 1
            if samples[kind] is None:
                samples[kind] = (name, entry.get("shape"), entry.get("dtype"))
        if idx % 100 == 0 or idx == limit:
            done = sum(totals.values()) / GiB
            print("    ... %d/%d shards, %.1f GiB accounted" % (idx, limit, done))

    if limit != len(shards):
        for kind in totals:
            totals[kind] = int(totals[kind] * scale)
    return totals, counts, samples, len(shards)


def kv_bytes_per_token(text_config, tp, kv_dtype_bytes):
    """Per-token KV bytes on one rank, for the full_attention layers only."""
    layer_types = text_config.get("layer_types")
    num_layers = text_config.get("num_hidden_layers", 0)
    if layer_types:
        n_full = sum(1 for t in layer_types if t == "full_attention")
        n_linear = sum(1 for t in layer_types if t == "linear_attention")
    else:
        n_full, n_linear = num_layers, 0

    num_kv_heads = text_config.get("num_key_value_heads")
    head_dim = text_config.get("head_dim")
    if not num_kv_heads or not head_dim:
        return None, n_full, n_linear
    kv_heads_local = max(1, num_kv_heads // tp)
    per_token = 2 * kv_heads_local * head_dim * kv_dtype_bytes * n_full
    return per_token, n_full, n_linear


def gdn_state_bytes(text_config, tp, n_linear, state_dtype_bytes):
    """Per-request GDN (linear attention) state bytes on one rank."""
    k_heads = text_config.get("linear_num_key_heads")
    v_heads = text_config.get("linear_num_value_heads")
    k_dim = text_config.get("linear_key_head_dim")
    v_dim = text_config.get("linear_value_head_dim")
    conv_kernel = text_config.get("linear_conv_kernel_dim", 4)
    if not all((k_heads, v_heads, k_dim, v_dim)):
        return None
    # mamba_utils.MambaStateShapeCalculator.gated_delta_net_state_shape
    conv_dim = k_dim * k_heads * 2 + v_dim * v_heads
    conv_state = (conv_dim // tp) * (conv_kernel - 1)
    temporal_state = (v_heads // tp) * v_dim * k_dim
    return (conv_state + temporal_state) * state_dtype_bytes * n_linear


def main(argv):
    opts = parse_args(argv)
    model_dir = Path(opts["model_path"])
    tp, dp = opts["tp"], opts["dp"]
    world = tp * dp
    ep = world if opts["ep"] else 1
    hbm = opts["hbm_gib"] * GiB
    util = opts["util"]

    print("model path      = " + str(model_dir))
    print("parallel plan   = TP%d x DP%d -> world %d rank(s), EP size %d"
          % (tp, dp, world, ep))
    print("hardware        = %d node(s) x %d NPU = %d card(s), %.1f GiB HBM each"
          % (opts["nodes"], opts["npus_per_node"], opts["nodes"] * opts["npus_per_node"],
             opts["hbm_gib"]))
    if opts["nodes"] * opts["npus_per_node"] != world:
        print("  NOTE: card count != world size; TP*DP must equal the cards actually used.")

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        print("[RED] missing " + str(config_path))
        return 1
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", config)

    collected = collect_weight_bytes(model_dir, opts["max_shards"])
    if collected is None:
        return 1
    totals, counts, samples, n_shards = collected

    disk_total = sum(totals.values())
    print("")
    print("checkpoint on disk: %.2f GiB across %d shard(s)" % (disk_total / GiB, n_shards))
    print("")
    print("  class        tensors        bytes   per-rank divisor   per-rank GiB")
    per_rank_weights = 0.0
    for kind, divisor, label in (
        ("expert", ep, "EP=%d" % ep),
        ("tp", tp, "TP=%d" % tp),
        ("replicated", 1, "1 (whole)"),
    ):
        share = totals[kind] / divisor
        per_rank_weights += share
        print("  %-11s %8d  %9.2f GiB  %-16s  %9.2f"
              % (kind, counts[kind], totals[kind] / GiB, label, share / GiB))
        if samples[kind]:
            name, shape, dtype = samples[kind]
            print("      e.g. %s  shape=%s dtype=%s" % (name, shape, dtype))

    print("")
    print("  => weights resident per rank: %.2f GiB" % (per_rank_weights / GiB))
    print("     (on-disk bytes; runtime may differ if a quant method repacks or")
    print("      up-casts a tensor -- check 'Loading model weights took X GB' in the log)")

    weight_scale = 1.0
    if opts["weights_gib"] is not None:
        measured_weights = opts["weights_gib"] * GiB
        weight_scale = measured_weights / per_rank_weights if per_rank_weights else 1.0
        print("     measured from the log     : %.2f GiB (x%.3f vs on-disk)"
              % (measured_weights / GiB, weight_scale))
        print("     using the measured value from here on; the overhead below is then a")
        print("     true activation+non-torch figure rather than a mixed residual.")
        per_rank_weights = measured_weights

    # ---- KV / GDN demand -------------------------------------------------
    kv_dtype_bytes = 1 if opts["kv_cache_dtype"] in ("fp8", "int8") else 2
    per_token, n_full, n_linear = kv_bytes_per_token(text_config, tp, kv_dtype_bytes)
    print("")
    print("layers: %d full_attention, %d linear_attention (GDN)" % (n_full, n_linear))
    kv_one_request = None
    if per_token is None:
        print("  [warn] config lacks num_key_value_heads/head_dim; cannot size KV")
    else:
        kv_one_request = per_token * opts["max_model_len"]
        print("  KV per token per rank      = %.1f KiB" % (per_token / 1024))
        print("  KV for one max_model_len=%d request = %.2f GiB"
              % (opts["max_model_len"], kv_one_request / GiB))
        print("  KV for max_num_seqs=%d full-length requests = %.2f GiB"
              % (opts["max_num_seqs"], kv_one_request * opts["max_num_seqs"] / GiB))
    gdn = gdn_state_bytes(text_config, tp, n_linear, 2)
    if gdn:
        print("  GDN state per request      = %.2f MiB (constant, not per token)"
              % (gdn / (1 << 20)))

    # ---- overhead: computed or back-solved -------------------------------
    print("")
    overhead = None
    if opts["observed_kv_gib"] is not None:
        obs_util = opts["observed_util"] if opts["observed_util"] is not None else util
        requested = hbm * obs_util
        non_kv = requested - opts["observed_kv_gib"] * GiB
        overhead = non_kv - per_rank_weights
        print("back-solving from the observed run (util=%.2f, available_kv=%.2f GiB):"
              % (obs_util, opts["observed_kv_gib"]))
        print("  requested        = %.2f GiB x %.2f = %.2f GiB"
              % (opts["hbm_gib"], obs_util, requested / GiB))
        print("  non_kv_cache     = requested - available_kv = %.2f GiB" % (non_kv / GiB))
        print("  weights (above)  = %.2f GiB" % (per_rank_weights / GiB))
        print("  => measured overhead (activation peak + HCCL + workspace + graph)")
        print("     = %.2f GiB per rank" % (overhead / GiB))
        if opts["weights_gib"] is None:
            print("  [note] weights here are the on-disk estimate, so this overhead also")
            print("         absorbs any load-time repack. Pass --weights-gib from the log's")
            print("         'Loading model weights took X GB' line to separate the two.")
        if overhead < 0:
            print("  [warn] negative overhead: the runtime weight footprint is LARGER than")
            print("         the on-disk bytes above. Suspect an up-cast at load time")
            print("         (a checkpoint that is not really quantized), or a tensor class")
            print("         this script mis-attributed. Compare with the log's")
            print("         'Loading model weights took X GB' line before trusting anything.")
    else:
        print("no --observed-kv-gib given; overhead (activation + HCCL + workspace)")
        print("cannot be computed statically. Re-run with the failing log's")
        print("  'Available KV cache memory: X GiB'  as --observed-kv-gib X")
        print("to turn that single number into a measured overhead.")

    # ---- verdict ---------------------------------------------------------
    print("")
    print("=" * 72)
    if overhead is None or kv_one_request is None:
        print("[RED] not enough inputs for a verdict "
              "(need --observed-kv-gib and a config with KV dims).")
        return 1

    budget = hbm * util
    need = per_rank_weights + overhead + kv_one_request
    print("per-rank budget at util=%.2f : %.2f GiB" % (util, budget / GiB))
    print("  weights                    : %.2f GiB" % (per_rank_weights / GiB))
    print("  overhead (measured)        : %.2f GiB" % (overhead / GiB))
    print("  KV floor (one full request): %.2f GiB" % (kv_one_request / GiB))
    print("  ---------------------------------------")
    print("  total needed               : %.2f GiB   (%s by %.2f GiB)"
          % (need / GiB, "FITS" if need <= budget else "OVER", abs(budget - need) / GiB))

    # What util would be required, and is it physically reachable?
    need_util = need / hbm
    print("")
    print("required gpu_memory_utilization = %.3f  (physical ceiling is < 1.0)" % need_util)

    # How many nodes would fix it, holding TP=8 and growing DP?
    print("")
    print("scaling (TP fixed at %d, DP grows; only expert weights shrink):" % tp)
    for nodes in (opts["nodes"], opts["nodes"] * 2, opts["nodes"] * 4):
        cards = nodes * opts["npus_per_node"]
        ep_n = cards if opts["ep"] else 1
        w = (totals["expert"] / ep_n + totals["tp"] / tp + totals["replicated"]) * weight_scale
        tot = w + overhead + kv_one_request
        print("  %2d node(s) / %3d card(s): weights %.2f + overhead %.2f + KV %.2f "
              "= %.2f GiB vs budget %.2f  -> %s"
              % (nodes, cards, w / GiB, overhead / GiB, kv_one_request / GiB,
                 tot / GiB, budget / GiB, "FITS" if tot <= budget else "OVER"))

    # Shorter contexts on the current hardware.
    print("")
    print("max_model_len sweep on the current %d card(s):" % (opts["nodes"] * opts["npus_per_node"]))
    headroom = budget - per_rank_weights - overhead
    print("  headroom left for KV = %.2f GiB" % (headroom / GiB))
    if headroom <= 0:
        print("  no positive headroom at any context length: weights + overhead alone")
        print("  already exceed the budget. Adding context length is not the lever.")
    else:
        max_tokens = int(headroom / per_token)
        print("  that is %d KV token(s) per rank, i.e. max_model_len up to %d"
              % (max_tokens, max_tokens))

    print("")
    if need <= budget:
        print("[GREEN] the current plan fits; the failure is elsewhere "
              "(check overhead knobs below anyway).")
        return 0

    print("[RED] the current plan does not fit. Levers, largest first:")
    if headroom <= 0:
        print("  * weights + overhead alone blow the budget -> more EP ranks (nodes)")
        print("    or a smaller/actually-quantized checkpoint is the ONLY fix.")
    print("  * --max-num-batched-tokens 16384 drives the activation peak; halving it")
    print("    is the cheapest overhead cut and needs no extra hardware.")
    print("  * HCCL_BUFFSIZE / HCCL_BUFFSIZE_EP (currently set in the launcher to")
    print("    1024 / 2048 MB) are non-torch memory on every rank.")
    print("  * --compilation-config cudagraph_mode: FULL_DECODE_ONLY reserves a graph")
    print("    pool; enforce-eager removes it at a throughput cost.")
    print("  * raising --gpu-memory-utilization only helps if the required util above")
    print("    is below ~0.95; %.3f is %s." % (need_util, "reachable" if need_util < 0.95 else "NOT reachable"))
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
