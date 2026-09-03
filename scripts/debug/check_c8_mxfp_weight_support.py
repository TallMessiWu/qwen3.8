#!/usr/bin/env python3
"""Decide whether a checkpoint can serve the C8-MXFP8 KV cache, and how good its V scales are.

Two questions, answered separately:

  1. Will C8 even turn on?  vllm_ascend/quantization/configs/modelslim_config.py
     enables it only when quant_model_description.json says
     kv_cache_type=K_DYNAMIC_V_STATIC_MXFP8_PER_CHANNEL.  Anything else and the
     run silently falls back to a bf16 cache served by FIA -- which, at head_dim
     256, then dies with EZ0010.  A missing key is therefore a hard RED.

  2. Are the V scales actually usable?  V is quantized with a *static*
     per-channel E8M0 scale read from the checkpoint (K stays dynamic).  Three
     ways that goes wrong, in descending order of severity:
       - a full-attention layer has no V scale at all: mxfp_c8.py falls back to
         an all-127 parameter, i.e. V is cached unscaled;
       - a channel's scale is 0: a minmax calibrator emits 0 when that channel's
         absmax was 0, and process_weights_after_loading() rewrites 0 to the
         neutral 127.  Harmless if the channel really is all-zero, precision
         loss if it is not.  This is the number to watch across checkpoints;
       - the element count does not divide by the TP size: _quant_weight_loader
         narrows by rank and then asserts on the size.

The name lookup mirrors modelslim_config.py's suffix_map exactly, so a GREEN
here means the framework really will find the scales -- both spellings are
accepted, because ModelSlim recipes changed the name and vLLM's own cache-scale
regex rewrites v_proj.v_scale to attn.v_scale before suffixes run.

Pure stdlib, read-only, never imports torch/vllm and never touches the NPU.
Only safetensors headers plus the V-scale tensors themselves are read, so this
is cheap even on a multi-hundred-GB checkpoint.

Usage: python3 check_c8_mxfp_weight_support.py [MODEL_PATH]
Final line is [GREEN] (exit 0) or [RED] (exit 1).
"""

import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_MODEL_PATH = "/mnt/share/weight/Qwen3.5-35B-A3B-mxfp4-c8"

REQUIRED_KV_CACHE_TYPE = "K_DYNAMIC_V_STATIC_MXFP8_PER_CHANNEL"

# modelslim_config.py maps both of these onto the attn.v_cache_scale parameter
# when enable_mxfp_c8_quant is on. Checkpoints use one or the other.
V_SCALE_SUFFIXES = (".v_proj.kv_cache_scale", ".v_proj.v_scale")

# K is quantized dynamically under this recipe, so a static K scale means the
# checkpoint was produced with a different one.
K_SCALE_SUFFIXES = (".k_proj.kv_cache_scale", ".k_proj.k_scale")

# TP sizes worth checking the per-channel split against.
TP_SIZES = (1, 2, 4, 8, 16)

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
MTP_RE = re.compile(r"(?:^|\.)mtp\.")


def read_safetensors_header(path):
    """Return (header dict, byte offset where the tensor payload starts)."""
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(header_len)), 8 + header_len


def read_u8_tensor(path, entry, data_start):
    """Read one small uint8 tensor's raw bytes; returns None for other dtypes."""
    if entry.get("dtype") != "U8":
        return None
    begin, end = entry["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(data_start + begin)
        return fh.read(end - begin)


def load_json(path):
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def layer_index(name):
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def collect_tensors(model_dir):
    """Map tensor name -> (shard path, header entry, payload offset)."""
    shards = sorted(model_dir.glob("*.safetensors"))
    tensors = {}
    for shard in shards:
        header, data_start = read_safetensors_header(shard)
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            tensors[name] = (shard, entry, data_start)
    return tensors, shards


def describe_scale(raw_bytes):
    """Summarize an E8M0 scale blob: (count, zeros, min, max, distinct)."""
    values = list(raw_bytes)
    if not values:
        return 0, 0, None, None, 0
    nonzero = [v for v in values if v != 0]
    return (
        len(values),
        len(values) - len(nonzero),
        min(values),
        max(values),
        len(set(values)),
    )


def check_kv_cache_type(model_path, failures):
    """Section 1: the flag modelslim_config.py actually reads."""
    print()
    print("--- 1. kv_cache_type (decides whether C8 is enabled) ---")
    desc = load_json(model_path / "quant_model_description.json")
    if desc is None:
        print("  quant_model_description.json: MISSING")
        failures.append("quant_model_description.json missing -> C8 never enables")
        return
    kv_cache_type = desc.get("kv_cache_type", "")
    print(f"  kv_cache_type = {kv_cache_type!r}")
    if kv_cache_type == REQUIRED_KV_CACHE_TYPE:
        print("  OK: matches what modelslim_config.py requires")
    else:
        print(f"  expected: {REQUIRED_KV_CACHE_TYPE!r}")
        failures.append(
            f"kv_cache_type != {REQUIRED_KV_CACHE_TYPE} -> C8 stays off, run falls back to bf16+FIA"
        )


def expected_channel_width(model_path):
    """Section 2: num_kv_heads * head_dim, or None when config.json is unclear."""
    print()
    print("--- 2. expected per-channel width ---")
    config = load_json(model_path / "config.json")
    text_config = {}
    if config:
        # Multimodal checkpoints nest the language model's config.
        for key in ("text_config", "language_config", "llm_config"):
            if isinstance(config.get(key), dict):
                text_config = config[key]
                break
    merged = {**(config or {}), **text_config}
    num_kv_heads = merged.get("num_key_value_heads")
    head_dim = merged.get("head_dim")
    if head_dim is None and merged.get("hidden_size") and merged.get("num_attention_heads"):
        head_dim = merged["hidden_size"] // merged["num_attention_heads"]
    if num_kv_heads and head_dim:
        width = num_kv_heads * head_dim
        print(f"  num_key_value_heads={num_kv_heads} head_dim={head_dim} -> expect {width} channels")
        return width
    print("  config.json did not give num_key_value_heads/head_dim; falling back to v_proj.weight shapes")
    return None


def scan_tensors(tensors):
    """Split the tensor names into the buckets the checks below need.

    Two of the buckets exist only to keep a miss honest: V_SCALE_SUFFIXES is a
    hard-coded pair, so a checkpoint that spells the scale differently would
    otherwise be reported as "no V scale" when it really means "named something
    this script does not know". attn_module_suffixes and scale_like_suffixes
    get printed on a miss so the real name is visible instead of guessed at.
    """
    found = {
        "attn_layers": set(),
        "mtp_attn_layers": set(),
        "v_scales": {},
        "mtp_v_scales": {},
        "k_scale_names": [],
        "v_proj_out_features": {},
        "attn_module_suffixes": Counter(),
        "scale_like_suffixes": Counter(),
    }
    for name, (shard, entry, data_start) in tensors.items():
        is_mtp = bool(MTP_RE.search(name))
        idx = layer_index(name)
        if idx is None:
            continue
        if ".self_attn." in name:
            found["attn_module_suffixes"][name.split(".self_attn.", 1)[1]] += 1
        if "scale" in name.lower() and not name.endswith(".weight_scale"):
            found["scale_like_suffixes"][".".join(name.rsplit(".", 2)[-2:])] += 1
        if name.endswith(".self_attn.v_proj.weight"):
            found["mtp_attn_layers" if is_mtp else "attn_layers"].add(idx)
            shape = entry.get("shape") or []
            if shape:
                found["v_proj_out_features"][(is_mtp, idx)] = shape[0]
        if any(name.endswith(suffix) for suffix in V_SCALE_SUFFIXES):
            found["mtp_v_scales" if is_mtp else "v_scales"][idx] = (name, shard, entry, data_start)
        if any(name.endswith(suffix) for suffix in K_SCALE_SUFFIXES):
            found["k_scale_names"].append(name)
    return found


def check_coverage(found, failures):
    """Section 3: does every full-attention layer carry a V scale?"""
    attn_layers = found["attn_layers"]
    v_scales = found["v_scales"]
    print(f"  full-attention layers (have self_attn.v_proj.weight): {len(attn_layers)} {sorted(attn_layers)}")
    if not attn_layers:
        print("  could not identify any full-attention layer -- naming may differ from Qwen3.5")
        failures.append("no self_attn.v_proj.weight found; cannot verify per-layer coverage")

    missing = sorted(attn_layers - set(v_scales))
    if missing:
        print(f"  layers WITHOUT a V scale: {missing}")
        failures.append(
            f"{len(missing)} full-attention layer(s) have no V scale -> those layers cache V unscaled (all-127)"
        )
        report_naming_candidates(found)
    elif attn_layers:
        print("  every full-attention layer has a V scale")

    extra = sorted(set(v_scales) - attn_layers)
    if extra:
        print(f"  V scales on layers with no v_proj.weight (ignored by the loader): {extra}")


def report_naming_candidates(found):
    """On a miss, show what IS there so "absent" can be told from "renamed".

    The suffix list this script matches on is hard-coded, so a checkpoint that
    ships the V-cache scale under a name nobody has seen yet would look exactly
    like one that has no scale at all. Printing the real inventory settles it.
    """
    print()
    print("  This script matches only: " + ", ".join("*" + s for s in V_SCALE_SUFFIXES))
    print("  What the checkpoint actually has, so a rename is not mistaken for an absence:")

    scale_like = found["scale_like_suffixes"]
    if scale_like:
        print("    scale-ish tensors (excluding *.weight_scale), by name suffix:")
        for suffix, count in sorted(scale_like.items(), key=lambda kv: (-kv[1], kv[0]))[:20]:
            print(f"      {suffix}  x{count}")
    else:
        print("    no scale-like tensors anywhere outside *.weight_scale")

    attn_suffixes = found["attn_module_suffixes"]
    if attn_suffixes:
        print("    everything under *.self_attn.:")
        for suffix, count in sorted(attn_suffixes.items()):
            print(f"      {suffix}  x{count}")

    print("    -> if a V-cache scale appears above under a different name, add it to")
    print("       V_SCALE_SUFFIXES here AND to suffix_map in modelslim_config.py; the")
    print("       framework matches the same hard-coded list and would drop it too.")


def check_scale_contents(found, expected_width, failures, warnings):
    """Section 4: shape, dtype, TP divisibility, and the zero-channel count."""
    print()
    print("--- 4. V scale contents (zeros are the precision risk) ---")
    v_scales = found["v_scales"]
    if not v_scales:
        print("  no V scales found at all")
        return

    name_styles = defaultdict(list)
    for idx, (name, _, _, _) in v_scales.items():
        suffix = next((s for s in V_SCALE_SUFFIXES if name.endswith(s)), "?")
        name_styles[suffix].append(idx)
    for suffix, idxs in sorted(name_styles.items()):
        print(f"  naming: *{suffix} on {len(idxs)} layer(s)")

    total_zeros = 0
    total_channels = 0
    rows = []
    for idx in sorted(v_scales):
        name, shard, entry, data_start = v_scales[idx]
        dtype = entry.get("dtype")
        shape = entry.get("shape") or []
        raw = read_u8_tensor(shard, entry, data_start)
        if raw is None:
            print(f"  layer {idx:<3} dtype={dtype} shape={shape} -- not uint8, cannot read as E8M0")
            failures.append(f"layer {idx} V scale has dtype {dtype}, expected U8")
            continue
        count, zeros, lo, hi, distinct = describe_scale(raw)
        total_zeros += zeros
        total_channels += count
        rows.append((idx, shape, count, zeros, lo, hi, distinct))

        width = found["v_proj_out_features"].get((False, idx))
        if width is not None and count != width:
            failures.append(f"layer {idx} V scale has {count} channels but v_proj.weight has {width} rows")
        if expected_width is not None and count != expected_width:
            failures.append(f"layer {idx} V scale has {count} channels, expected {expected_width}")

    header = f"  {'layer':<5} {'shape':<12} {'chans':<7} {'zeros':<7} {'min..max':<9} distinct"
    print(header)
    for idx, shape, count, zeros, lo, hi, distinct in rows:
        span = f"{lo}..{hi}"
        flag = "  <-- zeros" if zeros else ""
        print(f"  {idx:<5} {shape!s:<12} {count:<7} {zeros:<7} {span:<9} {distinct}{flag}")

    if not rows:
        return

    widths = {row[2] for row in rows}
    if len(widths) > 1:
        warnings.append(f"V scales disagree on channel count across layers: {sorted(widths)}")
    width = min(widths)
    bad_tp = [tp for tp in TP_SIZES if width % tp]
    if bad_tp:
        warnings.append(
            f"channel count {width} is not divisible by TP size(s) {bad_tp}; "
            "_quant_weight_loader would assert there"
        )

    print()
    pct = (100.0 * total_zeros / total_channels) if total_channels else 0.0
    print(f"  zero channels overall: {total_zeros} / {total_channels} ({pct:.2f}%)")
    print("    A zero means the calibrator saw absmax 0 on that channel; the loader rewrites it")
    print("    to the neutral 127, so V goes uncached-unscaled there. Fewer zeros is better.")
    layers_with_zeros = [row[0] for row in rows if row[3]]
    if layers_with_zeros:
        warnings.append(
            f"{len(layers_with_zeros)} layer(s) carry zero channels "
            f"({total_zeros} total, {pct:.2f}%): {layers_with_zeros}"
        )
    else:
        print("    none -- every channel has a real scale")


def check_k_scales(found, warnings):
    """Section 5: K_DYNAMIC means there should be no static K scale."""
    print()
    print("--- 5. K scales (recipe says K_DYNAMIC, so there should be none) ---")
    k_scale_names = found["k_scale_names"]
    if not k_scale_names:
        print("  none, as expected")
        return
    print(f"  found {len(k_scale_names)} static K scale tensor(s), e.g. {k_scale_names[0]}")
    warnings.append(
        f"checkpoint carries {len(k_scale_names)} static K scale(s); "
        "the C8-MXFP path quantizes K dynamically and ignores them"
    )


def check_mtp(found, warnings):
    """Section 6: the draft model's own V scales, which cost acceptance rate."""
    print()
    print("--- 6. MTP draft layers ---")
    mtp_attn_layers = found["mtp_attn_layers"]
    mtp_v_scales = found["mtp_v_scales"]
    if not mtp_attn_layers:
        print("  no MTP attention layers in this checkpoint (MTP either absent or unquantized)")
        return

    print(f"  MTP attention layers: {sorted(mtp_attn_layers)}")
    mtp_missing = sorted(mtp_attn_layers - set(mtp_v_scales))
    if mtp_missing:
        print(f"  MTP layers WITHOUT a V scale: {mtp_missing}")
        warnings.append(
            f"MTP layer(s) {mtp_missing} have no V scale -> the draft model caches V unscaled; "
            "costs acceptance rate but does not error (the target verifies)"
        )
        return

    print("  every MTP attention layer has a V scale")
    for idx in sorted(mtp_v_scales):
        _, shard, entry, data_start = mtp_v_scales[idx]
        raw = read_u8_tensor(shard, entry, data_start)
        if raw is not None:
            count, zeros, lo, hi, distinct = describe_scale(raw)
            print(f"    mtp layer {idx}: {count} chans, {zeros} zeros, {lo}..{hi}, distinct={distinct}")


def main():
    model_path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH)
    print("=== C8-MXFP8 KV cache support check ===")
    print("model path: " + str(model_path))

    if not model_path.is_dir():
        print("[RED] model path is not a directory")
        return 1

    failures = []
    warnings = []

    check_kv_cache_type(model_path, failures)
    expected_width = expected_channel_width(model_path)

    print()
    print("--- 3. per-layer V scale ---")
    tensors, shards = collect_tensors(model_path)
    if not tensors:
        print("  no *.safetensors found under the model path")
        print()
        print("  [FAIL] no safetensors shards found")
        print()
        print("[RED]")
        return 1
    print(f"  read {len(shards)} shard header(s), {len(tensors)} tensors")

    found = scan_tensors(tensors)
    check_coverage(found, failures)
    check_scale_contents(found, expected_width, failures, warnings)
    check_k_scales(found, warnings)
    check_mtp(found, warnings)

    print()
    print("=== verdict ===")
    for warning in warnings:
        print("  [WARN] " + warning)
    for failure in failures:
        print("  [FAIL] " + failure)

    if failures:
        print()
        print("[RED]")
        return 1
    print()
    print("  C8 will enable, and every full-attention layer's V scale will be found and loaded.")
    print("[GREEN]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
