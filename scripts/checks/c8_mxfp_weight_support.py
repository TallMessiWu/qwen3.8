#!/usr/bin/env python3
"""Decide whether a checkpoint can serve the C8-MXFP8 KV cache, and how good its V scales are.

Two questions, answered separately:

  1. Will C8 even turn on?  vllm_ascend/quantization/configs/modelslim_config.py
     enables it only when quant_model_description.json says
     kv_cache_type=K_DYNAMIC_V_STATIC_MXFP8_PER_CHANNEL.  Anything else and the
     run silently falls back to a bf16 cache served by FIA -- which, at head_dim
     256, then dies with EZ0010.  A missing key is therefore a hard RED.
     Three KV-cache recipes share that file and only this one is readable by
     QFA, so the report names which recipe the checkpoint actually selected:
     one carrying FAKQuant's fa_k/fa_v scale+offset, or int8 C8, *did* quantize
     its KV cache, just into a layout QFA cannot use.  Saying "unquantized"
     there would be wrong, and the difference decides what to do about it.

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

Usage: python3 c8_mxfp_weight_support.py [MODEL_PATH]
Final line is [GREEN] (exit 0) or [RED] (exit 1).
"""

import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_MODEL_PATH = "/mnt/share/weight/Qwen3.5-35B-A3B-mxfp4-c8"

# Two ways a checkpoint can ask for the MXFP8 KV cache. The older recipe puts
# one key at the top of quant_model_description.json; the newer one drops that
# key and tags each attention layer instead. Both describe the same thing --
# K quantized dynamically, V carrying a static per-channel scale.
REQUIRED_KV_CACHE_TYPE = "K_DYNAMIC_V_STATIC_MXFP8_PER_CHANNEL"
PER_LAYER_QUANT_TYPES = ("QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL",)
LAYER_QUANT_TYPE_SUFFIX = ".self_attn.quant_type"

# All map onto the attn.v_cache_scale parameter. Checkpoints use one of them;
# fa_v.scale is the newer spelling and reuses a name FAQuant also uses, so the
# tensor's own shape and dtype are what tell the two recipes apart, not the name.
V_SCALE_SUFFIXES = (".v_proj.kv_cache_scale", ".v_proj.v_scale", ".fa_v.scale")

# K is quantized dynamically under this recipe, so a static K scale means the
# checkpoint was produced with a different one.
K_SCALE_SUFFIXES = (".k_proj.kv_cache_scale", ".k_proj.k_scale", ".fa_k.scale")

# Mirrors MXFP_KV_SCALE_GROUP_SIZE in vllm_ascend/device/mxfp_kv_cache.py:
# the K scale cache groups head_dim into 64-wide slots, and building it
# raises outright when head_dim does not divide by this.
SCALE_GROUP_SIZE = 64

# TP sizes worth checking the per-channel split against. Goes past 16 because
# 397B-class models are served far wider than the 35B this started on.
TP_SIZES = (1, 2, 4, 8, 16, 32, 64)

# head_dim values QFA has actually been run at here. Others are not known to
# fail -- they are just untested, and the report says so rather than implying
# either answer.
MEASURED_HEAD_DIMS = (256,)

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


def check_kv_cache_type(model_path, failures, warnings):
    """Section 1: which KV-cache quantization recipe the framework will pick.

    Several recipes share modelslim_config.py and only the MXFP8 one is readable
    by QFA, so name the one the description actually selects rather than only
    saying "not MXFP8". Two spellings ask for MXFP8:
      kv_cache_type=K_DYNAMIC_V_STATIC_MXFP8_PER_CHANNEL   one key, at the top
      <layer>.self_attn.quant_type=QK_MXFP8_DYNAMIC_...    one key per layer
    and two ask for something else:
      kv_cache_type=C8            kv_c8.py, int8 dense C8
      fa_quant_type=<non-empty>   kv_c8.py FAKQuant, which QFA cannot read
    """
    print()
    print("--- 1. which KV-cache recipe does the description select? ---")
    desc = load_json(model_path / "quant_model_description.json")
    if desc is None:
        print("  quant_model_description.json: MISSING")
        failures.append("quant_model_description.json missing -> C8 never enables")
        return

    kv_cache_type = desc.get("kv_cache_type", "")
    fa_quant_type = desc.get("fa_quant_type", "")
    print(f"  top-level kv_cache_type = {kv_cache_type!r}")
    print(f"  top-level fa_quant_type = {fa_quant_type!r}")

    # Per-layer tags: collect every value so a checkpoint that mixes recipes
    # across layers is visible rather than averaged away.
    per_layer = defaultdict(list)
    for key, value in desc.items():
        if key.endswith(LAYER_QUANT_TYPE_SUFFIX):
            idx = layer_index(key)
            per_layer[str(value)].append(idx)
    if per_layer:
        print(f"  per-layer *{LAYER_QUANT_TYPE_SUFFIX}:")
        for value, idxs in sorted(per_layer.items()):
            known = " (MXFP8 C8)" if value in PER_LAYER_QUANT_TYPES else ""
            shown = sorted(i for i in idxs if i is not None)
            print(f"    {value!r} x{len(idxs)}{known}  layers {shown}")
    else:
        print(f"  per-layer *{LAYER_QUANT_TYPE_SUFFIX}: none")

    mxfp_layers = sorted(
        i for value, idxs in per_layer.items() if value in PER_LAYER_QUANT_TYPES for i in idxs if i is not None
    )
    by_top = kv_cache_type == REQUIRED_KV_CACHE_TYPE
    if by_top or mxfp_layers:
        how = "top-level kv_cache_type" if by_top else f"{len(mxfp_layers)} per-layer quant_type tag(s)"
        print(f"  OK: selects the MXFP8 C8 path (mxfp_c8.py) via {how} -- the one QFA can read")
        # The framework has one global switch, so a checkpoint tagging only some
        # layers would turn C8 on for all of them.
        other = sorted(
            i for value, idxs in per_layer.items() if value not in PER_LAYER_QUANT_TYPES for i in idxs if i is not None
        )
        if other:
            warnings.append(
                f"layers {other} carry a different quant_type than {mxfp_layers}; "
                "enable_mxfp_c8_quant is global, so C8 would apply to all of them alike"
            )
        return

    print(f"  expected: kv_cache_type == {REQUIRED_KV_CACHE_TYPE!r}, "
          f"or a per-layer quant_type in {PER_LAYER_QUANT_TYPES}")
    failures.append("no MXFP8 recipe selected -> C8 stays off, run falls back to bf16+FIA")
    if fa_quant_type:
        print("  note: fa_quant_type is set, so enable_fa_quant wins instead. That is the")
        print("        FAKQuant path in kv_c8.py, which on A5 caches K/V as float8_e4m3 with a")
        print("        static per-channel scale+offset -- a different cache layout from MXFP8's")
        print("        per-32-element E8M0 scale planes. QFA cannot read it.")
    elif kv_cache_type == "C8":
        print("  note: this selects the int8 dense-attention C8 path (kv_c8.py), not MXFP8.")


def expected_channel_width(model_path, failures, warnings):
    """Section 2: geometry the C8 cache needs, and what would keep QFA off anyway.

    Two separate gates live here. The cache itself refuses to be built unless
    head_dim divides by 64 (validate_mxfp_k_scale_head_dim). And even with the
    cache up, _qfa_serves declines any layer that has attention sinks or a
    sliding window, so a checkpoint carrying either would quietly fall back to
    FIA on those layers -- worth knowing before blaming the operator.
    """
    print()
    print("--- 2. model geometry and QFA's serve conditions ---")
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

    width = None
    if num_kv_heads and head_dim:
        width = num_kv_heads * head_dim
        print(f"  num_key_value_heads={num_kv_heads} head_dim={head_dim} -> expect {width} channels")

        # Hard gate: the K scale cache groups along head_dim in 64-wide slots.
        if head_dim % SCALE_GROUP_SIZE:
            failures.append(
                f"head_dim {head_dim} is not divisible by {SCALE_GROUP_SIZE}; "
                "validate_mxfp_k_scale_head_dim() raises and the C8 cache cannot be built"
            )
        else:
            print(f"  head_dim % {SCALE_GROUP_SIZE} == 0, so the K scale cache is constructible")

        # TP splits the per-channel scale; _quant_weight_loader asserts on a
        # ragged split. 397B-class models run wide, so check past 16.
        bad_tp = [tp for tp in TP_SIZES if width % tp]
        if bad_tp:
            warnings.append(
                f"{width} channels is not divisible by TP size(s) {bad_tp}; "
                "_quant_weight_loader would assert if served at those"
            )
        else:
            print(f"  {width} channels divides evenly across TP {list(TP_SIZES)}")

        if head_dim not in MEASURED_HEAD_DIMS:
            warnings.append(
                f"head_dim {head_dim} has not been exercised on QFA here (measured: "
                f"{sorted(MEASURED_HEAD_DIMS)}); the operator may well take it, but that is untested"
            )
    else:
        print("  config.json did not give num_key_value_heads/head_dim; falling back to v_proj.weight shapes")

    # Soft gates: these do not stop C8, they stop QFA from serving those layers.
    sliding = merged.get("sliding_window")
    uses_sliding = merged.get("use_sliding_window", sliding is not None)
    if sliding and uses_sliding:
        warnings.append(
            f"config sets sliding_window={sliding} with use_sliding_window={uses_sliding}; "
            "_qfa_serves declines any layer with a sliding window, which falls back to FIA"
        )
    layer_types = merged.get("layer_types")
    if isinstance(layer_types, list) and any("sliding" in str(t) for t in layer_types):
        sliding_layers = sum(1 for t in layer_types if "sliding" in str(t))
        warnings.append(f"layer_types marks {sliding_layers} sliding-attention layer(s); QFA will not serve those")
    if merged.get("attention_sinks") or merged.get("sinks"):
        warnings.append("config declares attention sinks; _qfa_serves declines those layers")

    return width


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
        "fa_offsets": {},
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
        if name.endswith(".offset") and ".fa_" in name:
            found["fa_offsets"][idx] = (name, shard, entry, data_start)
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
        if any(s.startswith(("fa_k.", "fa_v.", "fa_q.")) for s in scale_like):
            print()
            print("    ^ those fa_* names are the FAKQuant recipe (kv_c8.py), NOT MXFP8 C8.")
            print("      Tell them apart by the offset: MXFP8 scales are powers of two (E8M0)")
            print("      and need none, and its recipe is K_DYNAMIC so only V carries a static")
            print("      scale. A checkpoint with fa_k/fa_v scale+offset did quantize its KV")
            print("      cache -- just into a layout QFA cannot read. It is not 'unquantized'.")
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


def analyse_scale_packing(raw, count, expected):
    """When the scale is short by an exact factor, say whether the bytes are packed.

    Two readings fit a tensor with half the channels it should have: each byte
    holds two 4-bit values, or one exponent is shared between two channels. The
    byte pattern separates them. An E8M0 exponent for attention V lands in a
    narrow band (roughly 115-127, i.e. 2^-12..2^0), so a per-channel scale shows
    few distinct values clustered together and a near-constant high nibble. Two
    independent 4-bit fields instead spread bytes across 0..255.

    Worth checking rather than assuming: MX packs two FP4 *data* elements per
    byte, but its scale is always a full E8M0 byte -- so a packed scale would be
    a departure from the standard, not an application of it.
    """
    if raw is None or not count or expected != count * 2:
        return
    values = list(raw)
    distinct = sorted(set(values))
    high = sorted({v >> 4 for v in values})
    low = sorted({v & 0xF for v in values})
    print()
    print(f"    channel count is exactly half of {expected}; is each byte packing two values?")
    print(f"      distinct bytes ({len(distinct)}): {distinct[:12]}{'...' if len(distinct) > 12 else ''}")
    print(f"      high nibbles: {high}    low nibbles: {low}")
    as_exponents = [v - 127 for v in (distinct[0], distinct[-1])]
    print(f"      read as E8M0: 2^{as_exponents[0]} .. 2^{as_exponents[1]}")
    if len(high) == 1 and max(distinct) - min(distinct) < 32:
        print("      -> NOT packed: the high nibble never varies and the bytes sit in a narrow")
        print("         band, which is what one E8M0 exponent per channel looks like. The two")
        print("         KV heads most likely share one set of per-head_dim scales.")
    else:
        print("      -> bytes are spread out and both nibbles vary, so packing is possible;")
        print("         confirm the intended layout with whoever exported the checkpoint.")


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
    mismatched = []
    first_raw = None
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
        if first_raw is None:
            first_raw = raw
        total_zeros += zeros
        total_channels += count
        rows.append((idx, shape, count, zeros, lo, hi, distinct))

        width = found["v_proj_out_features"].get((False, idx))
        if width is not None and count != width:
            mismatched.append((idx, count, width))
        elif expected_width is not None and count != expected_width:
            mismatched.append((idx, count, expected_width))

    header = f"  {'layer':<5} {'shape':<12} {'chans':<7} {'zeros':<7} {'min..max':<9} distinct"
    print(header)
    for idx, shape, count, zeros, lo, hi, distinct in rows:
        span = f"{lo}..{hi}"
        flag = "  <-- zeros" if zeros else ""
        print(f"  {idx:<5} {shape!s:<12} {count:<7} {zeros:<7} {span:<9} {distinct}{flag}")

    if mismatched:
        counts = {c for _, c, _ in mismatched}
        wants = {w for _, _, w in mismatched}
        layers = [i for i, _, _ in mismatched]
        failures.append(
            f"{len(mismatched)} layer(s) have {sorted(counts)} V-scale channels but "
            f"{sorted(wants)} are expected ({len(layers)} layers: {layers[:6]}"
            f"{'...' if len(layers) > 6 else ''}) -- _quant_weight_loader asserts on this"
        )
        analyse_scale_packing(first_raw, min(counts), min(wants))

    if not rows:
        return

    widths = {row[2] for row in rows}
    if len(widths) > 1:
        warnings.append(f"V scales disagree on channel count across layers: {sorted(widths)}")
    width = min(widths)
    # Section 2 already reports TP divisibility from config.json; only fall back
    # to the measured width when config could not supply the geometry.
    if expected_width is None:
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
    if any(".fa_k." in n for n in k_scale_names):
        # kvcache_quant_layers is collected from fa_k.scale, and is_fa_quant_layer
        # gates a branch that sits *before* enable_mxfp_c8_quant in
        # get_quant_method -- so these would take the layer away from MXFP8.
        warnings.append(
            f"{len(k_scale_names)} fa_k.scale tensor(s) present: with fa_quant_type also set "
            "these populate kvcache_quant_layers, and the FAKQuant branch is checked before "
            "the MXFP8 one in get_quant_method, so those layers would leave the MXFP8 path"
        )
    else:
        warnings.append(
            f"checkpoint carries {len(k_scale_names)} static K scale(s); "
            "the C8-MXFP path quantizes K dynamically and ignores them"
        )


def check_offsets(found, warnings):
    """Report any fa_*.offset tensors: MXFP8 has no use for them.

    An E8M0 scale is a power of two applied symmetrically, so the MXFP8 path
    neither loads nor needs an offset. A checkpoint that ships one is either
    carrying a leftover from the affine recipe or expecting asymmetric
    dequantization -- and those differ only in whether the values are zero,
    which is worth knowing before deciding to ignore them.
    """
    print()
    print("--- 5b. fa_*.offset tensors (MXFP8 has no offset) ---")
    offsets = found["fa_offsets"]
    if not offsets:
        print("  none")
        return
    all_zero = True
    for idx in sorted(offsets):
        name, shard, entry, data_start = offsets[idx]
        raw = read_u8_tensor(shard, entry, data_start)
        shape, dtype = entry.get("shape"), entry.get("dtype")
        if raw is None:
            print(f"  layer {idx:<3} {name.rsplit('.', 2)[-2]}.offset  shape={shape} dtype={dtype} (not uint8)")
            all_zero = False
            continue
        nonzero = sum(1 for b in raw if b != 0)
        all_zero &= nonzero == 0
        print(f"  layer {idx:<3} shape={shape} dtype={dtype}  non-zero bytes: {nonzero}/{len(raw)}")
    if all_zero:
        print("  -> all zero, so ignoring them costs nothing")
    else:
        warnings.append(
            "fa_*.offset carries non-zero values; the MXFP8 path applies scale only, "
            "so those offsets would be silently dropped -- check what the exporter meant by them"
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

    check_kv_cache_type(model_path, failures, warnings)
    expected_width = expected_channel_width(model_path, failures, warnings)

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
    check_offsets(found, warnings)
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
