#!/usr/bin/env python3
"""Replay vllm-ascend PR #16051's fused-shard check against a ModelSlim checkpoint.

The check itself is not new -- "all shards of a fused layer must carry the same
quant type" predates the PR.  What the PR changed is *which* fused groups get
checked and *which* keys get looked up, and both changes land on Qwen3.5/3.8:

  1. The mapping source moved.  vllm-ascend used to hardcode
     packed_modules_model_mapping[model_type] for 30+ models; the PR deletes
     that table and takes vLLM's own model-class packed_modules_mapping by
     reference (model_loader/utils.py: configure_quant_config), keeping only
     six upstream gaps in UPDATED_PACKED_MODULES_MAPPING.  "qwen3_5_moe" is
     NOT one of the six, so 397B is now checked entirely against
     Qwen3_5MoeForConditionalGeneration.packed_modules_mapping -- which adds
     the vision tower's "qkv" entry the old ascend table never had.
  2. The expert shard names are auto-discovered from the checkpoint instead of
     hardcoded to gate/up/down, so whatever the description actually names
     under experts.<i>. becomes mandatory for every expert-0 lookup.
  3. Shard keys are now built with prefix.removesuffix(proj_name) instead of
     prefix.replace(proj_name, ...).  The old form also rewrote earlier
     occurrences of proj_name inside the path, so some lookups used to miss the
     intended key; the check now really lands where it was meant to.

Two ways a checkpoint dies at model-init time, both replayed verbatim here from
get_quant_type_for_layer (vllm_ascend/quantization/configs/modelslim_config.py):

  MISSING -> KeyError.  quant_description[shard_key] is a bare dict index.  A
     fused module the model builds whose shards are absent from the description
     raises, even though a *non*-fused module with no entry is silently treated
     as unquantized (post-PR it uses .get()).  Only Gemma4's replicated v_proj
     is allowed to be absent.
  MIXED -> ValueError, "Not all shards of X are quantized with same quant
     type".  Exact string equality, so W4A4 next to W8A8_DYNAMIC fails, and so
     does W4A4 next to FLOAT.  This is the one to watch on a *_multi recipe:
     mixing per layer is fine, mixing inside one fused group is fatal.

Also reported, because the replayed check cannot see them:

  - Expert disagreement beyond expert 0.  The fused prefix is ".mlp.experts",
    so the mapping only ever names experts.0.*.  Experts 1..N-1 are never
    compared.  A checkpoint that quantized expert 0 differently from the rest
    passes this check and breaks later, so every expert is scanned separately.
  - The MTP drafter's mapping (Qwen3_5MoeMTP) carries no in_proj_qkvz /
    in_proj_ba, so a GDN draft layer resolves through the non-packed branch and
    comes out unquantized instead of raising.  Reported as a warning.

Pure stdlib, read-only.  Reads only config.json and quant_model_description.json
-- no safetensors, no torch, no vllm, and nothing that touches an NPU.  Pass
--from-vllm to resolve packed_modules_mapping by importing the real vLLM model
class instead of the table baked in below (slower, imports torch).

Usage:
    python3 packed_shard_quant_uniformity.py [MODEL_PATH]
    python3 packed_shard_quant_uniformity.py --desc /path/quant_model_description.json \
                                             --config /path/config.json
Final line is [GREEN] (exit 0) or [RED] (exit 1).
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_MODEL_PATH = "/mnt/share/weight/qwen3.5-397b-w4a4_multi"

MAX_EXAMPLES = 8

# Copied from vllm_ascend/quantization/configs/modelslim_config.py after PR
# #16051.  Only these six model types still get a vllm-ascend-side override;
# everything else, qwen3_5_moe included, relies on the vLLM model class.
UPDATED_PACKED_MODULES_MAPPING: dict[str, dict[str, list[str]]] = {
    "glm5_next": {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "fused_qkvbfg_a_proj": ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"],
    },
    "deepseek_mtp": {"gate_up_proj": ["gate_proj", "up_proj"]},
    "pangu_ultra_moe_mtp": {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    },
    "qwen3_vl_moe": {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    },
    "longcat_flash": {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    },
    "bailing_hybrid": {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "o_proj": ["dense"],
    },
    "step3p5_mtp": {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    },
}

# Transcribed from vLLM 0.27.1, the version the server runs:
#   qwen3_vl.py:1713   Qwen3VLForConditionalGeneration.packed_modules_mapping
#   qwen3_5.py:294     Qwen3_5ForCausalLMBase.packed_modules_mapping
#   qwen3_5.py:447     Qwen3_5ForConditionalGeneration = VL mapping | GDN pair
#   qwen3_5_mtp.py:213 Qwen3_5MTP.packed_modules_mapping (Qwen3_5MoeMTP inherits)
# The vLLM module is the fused in_proj_qkvz / in_proj_ba; the checkpoint stores
# in_proj_qkv / in_proj_z / in_proj_b / in_proj_a separately, which is why the
# mapping splits them.  Re-check these against the installed vLLM with
# --from-vllm whenever the vendor moves.
_QWEN3_5_GDN = {
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}
BUILTIN_PACKED_MODULES_MAPPING: dict[str, dict[str, list[str]]] = {
    "Qwen3_5MoeForConditionalGeneration": {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "qkv": ["qkv"],
        **_QWEN3_5_GDN,
    },
    "Qwen3_5MoeForCausalLM": {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        **_QWEN3_5_GDN,
    },
    "Qwen3_5MoeMTP": {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    },
}
BUILTIN_PACKED_MODULES_MAPPING["Qwen3_5ForConditionalGeneration"] = BUILTIN_PACKED_MODULES_MAPPING[
    "Qwen3_5MoeForConditionalGeneration"
]
BUILTIN_PACKED_MODULES_MAPPING["Qwen3_5ForCausalLM"] = BUILTIN_PACKED_MODULES_MAPPING["Qwen3_5MoeForCausalLM"]
BUILTIN_PACKED_MODULES_MAPPING["Qwen3_5MTP"] = BUILTIN_PACKED_MODULES_MAPPING["Qwen3_5MoeMTP"]


# --------------------------------------------------------------------------
# verbatim from vllm_ascend/quantization/configs/modelslim_config.py @ #16051
# --------------------------------------------------------------------------
def _is_missing_v_shard(shard_key: str, quant_description: dict) -> bool:
    if not shard_key.endswith(".v_proj.weight"):
        return False
    shard_prefix = shard_key[: -len("v_proj.weight")]
    return f"{shard_prefix}q_proj.weight" in quant_description and f"{shard_prefix}k_proj.weight" in quant_description


def get_quant_type_for_layer(quant_description, prefix, packed_modules_mapping=None):
    if packed_modules_mapping is None:
        packed_modules_mapping = dict()  # noqa: C408 - verbatim from modelslim_config.py
    proj_name = prefix.split(".")[-1]
    if proj_name in packed_modules_mapping:
        quant_type = None
        shard_prefixes = [
            prefix.removesuffix(proj_name) + shard_proj_name for shard_proj_name in packed_modules_mapping[proj_name]
        ]
        for shard_prefix in shard_prefixes:
            shard_key = shard_prefix + ".weight"
            if shard_key not in quant_description and _is_missing_v_shard(shard_key, quant_description):
                continue
            shard_quant_type = quant_description[shard_key]
            if quant_type is None:
                quant_type = shard_quant_type
            elif shard_quant_type != quant_type:
                raise ValueError(
                    f"Not all shards of {prefix} are quantized with same quant type. "
                    f"Shard {proj_name} uses {shard_quant_type}, but another shard "
                    f"uses {quant_type}. Please check quantization config."
                )
    else:
        quant_type = quant_description.get(prefix + ".weight")
    return quant_type if quant_type != "FLOAT" else None


# Pre-PR pair, kept so the report can say whether #16051 changed the outcome for
# a given prefix or merely inherited an already-broken checkpoint.
def _pre_pr_verdict(quant_description, prefix, packed_modules_mapping):
    """Return ("ok"|"mixed"|"missing", detail) under the pre-#16051 logic."""
    proj_name = prefix.split(".")[-1]
    if proj_name in packed_modules_mapping:
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name) for shard_proj_name in packed_modules_mapping[proj_name]
        ]
        is_skipped = None
        for shard_prefix in shard_prefixes:
            shard_key = shard_prefix + ".weight"
            if shard_key not in quant_description and _is_missing_v_shard(shard_key, quant_description):
                continue
            if shard_key not in quant_description:
                return "missing", shard_key
            is_shard_skipped = quant_description[shard_key] == "FLOAT"
            if is_skipped is None:
                is_skipped = is_shard_skipped
            elif is_shard_skipped != is_skipped:
                return "mixed", "FLOAT mixed with non-FLOAT"
        if is_skipped:
            return "ok", "skipped (all FLOAT)"
        # Not skipped -> get_linear_quant_type ran the same strict comparison.
        types = {
            quant_description[shard_prefix + ".weight"]
            for shard_prefix in shard_prefixes
            if shard_prefix + ".weight" in quant_description
        }
        if len(types) > 1:
            return "mixed", "strict type mismatch"
        return "ok", next(iter(types), "?")
    return "ok", "not a fused group"


# --------------------------------------------------------------------------
def load_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"  cannot read {path}: {exc}")
        return None


def apply_extra_quant_adaptations(desc):
    """Mirror AscendModelSlimConfig._apply_extra_quant_adaptations.

    Only the aliasing that can decide presence matters here: weight_packed ->
    weight (compressed-tensors style descriptions would otherwise look like
    every shard is missing) and the shared_head rewrites used by MTP blocks.
    """
    desc = dict(desc)
    extra = {}
    for key in desc:
        if "shared_head" in key:
            extra[key.replace(".shared_head.", ".")] = desc[key]
        if "transformer.shared_head.output." in key:
            extra[key.replace("transformer.shared_head.output.", "shared_head.head.")] = desc[key]
        if "transformer.shared_head.norm." in key:
            extra[key.replace("transformer.shared_head.norm.", "shared_head.norm.")] = desc[key]
        if "weight_packed" in key:
            extra[key.replace("weight_packed", "weight")] = desc[key]
    desc.update(extra)
    return desc


def resolve_packed_mapping(architecture, model_type, desc, from_vllm):
    """Rebuild what AscendModelSlimConfig.packed_modules_mapping ends up holding."""
    mapping = None
    source = ""
    if from_vllm:
        mapping, source = _mapping_from_vllm(architecture)
        if mapping is not None:
            _compare_with_builtin(architecture, mapping)
    if mapping is None:
        builtin = BUILTIN_PACKED_MODULES_MAPPING.get(architecture)
        if builtin is None:
            print(f"  no built-in mapping for architecture {architecture!r}")
            print("  re-run with --from-vllm, or add the architecture to BUILTIN_PACKED_MODULES_MAPPING")
            return None, ""
        mapping = {name: list(shards) for name, shards in builtin.items()}
        source = f"built-in table (vLLM 0.27.1 source) for {architecture}"

    # _update_packed_modules_mapping: vllm-ascend deltas, then expert discovery.
    if model_type in UPDATED_PACKED_MODULES_MAPPING:
        mapping.update(UPDATED_PACKED_MODULES_MAPPING[model_type])
        source += f" + UPDATED_PACKED_MODULES_MAPPING[{model_type}]"
    if "experts" not in mapping:
        shard_names = set()
        for key in desc:
            match = re.search(r"\.experts\.\d+\.(\w+)\.weight$", key)
            if match:
                shard_names.add(match.group(1))
        if shard_names:
            mapping["experts"] = [f"experts.0.{name}" for name in sorted(shard_names)]
    return mapping, source


def _compare_with_builtin(architecture, mapping):
    """Say whether the table baked into this script still matches the real class.

    The built-in table was transcribed from vLLM 0.27.1.  When the vendor moves,
    this is the line that tells you the transcription went stale -- and a new
    fused group appearing here means a new way for the check to fire.
    """
    builtin = BUILTIN_PACKED_MODULES_MAPPING.get(architecture)
    if builtin is None:
        print(f"  note: no built-in table for {architecture} to compare against")
        return
    only_vllm = sorted(set(mapping) - set(builtin))
    only_builtin = sorted(set(builtin) - set(mapping))
    differing = sorted(name for name in set(mapping) & set(builtin) if list(mapping[name]) != list(builtin[name]))
    if not (only_vllm or only_builtin or differing):
        print("  built-in table matches the installed vLLM class exactly")
        return
    print("  built-in table is STALE against the installed vLLM class:")
    for name in only_vllm:
        print(f"    only in vLLM     : {name} <- {mapping[name]}")
    for name in only_builtin:
        print(f"    only in built-in : {name} <- {builtin[name]}")
    for name in differing:
        print(f"    shards differ    : {name} vLLM={mapping[name]} built-in={builtin[name]}")
    print("  the verdict below uses the vLLM values; update BUILTIN_PACKED_MODULES_MAPPING")


def _mapping_from_vllm(architecture):
    try:
        from vllm.model_executor.models.registry import ModelRegistry

        model_cls = ModelRegistry._try_load_model_cls(architecture)
        if model_cls is None:
            print(f"  vLLM registry has no entry for {architecture!r}")
            return None, ""
        mapping = getattr(model_cls, "packed_modules_mapping", None)
        if mapping is None:
            print(f"  {architecture} has no packed_modules_mapping attribute")
            return None, ""
        return {name: list(shards) for name, shards in mapping.items()}, f"imported {architecture} from vLLM"
    except Exception as exc:  # noqa: BLE001 - any import failure falls back
        print(f"  --from-vllm failed ({type(exc).__name__}: {exc}); falling back to the built-in table")
        return None, ""


def detect_layer_namespace(desc):
    """Return the prefix that precedes 'layers.<i>.' in the description keys."""
    counts = Counter()
    for key in desc:
        match = re.match(r"^(.*?)layers\.\d+\.", key)
        if match:
            counts[match.group(1)] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def expected_prefixes_from_config(text_config, namespace, mapping):
    """Fused modules the Qwen3.5 hybrid MoE model actually constructs.

    Mirrors Qwen3_5DecoderLayer: full_attention layers build self_attn.qkv_proj,
    linear_attention layers build linear_attn.in_proj_qkvz and .in_proj_ba, and
    every layer builds the MoE block (mlp_only_layers is empty on 397B).
    """
    prefixes = []
    layer_types = text_config.get("layer_types") or []
    num_layers = text_config.get("num_hidden_layers") or len(layer_types)
    mlp_only = set(text_config.get("mlp_only_layers") or [])
    has_experts = "experts" in mapping
    has_shared = bool(text_config.get("shared_expert_intermediate_size"))
    for idx in range(num_layers):
        base = f"{namespace}layers.{idx}."
        layer_type = layer_types[idx] if idx < len(layer_types) else "full_attention"
        if layer_type == "full_attention":
            prefixes.append((base + "self_attn.qkv_proj", f"layer {idx} full_attention"))
        else:
            prefixes.append((base + "linear_attn.in_proj_qkvz", f"layer {idx} linear_attention"))
            prefixes.append((base + "linear_attn.in_proj_ba", f"layer {idx} linear_attention"))
        if idx in mlp_only:
            prefixes.append((base + "mlp.gate_up_proj", f"layer {idx} dense mlp"))
            continue
        if has_experts:
            prefixes.append((base + "mlp.experts", f"layer {idx} moe"))
        if has_shared:
            prefixes.append((base + "mlp.shared_expert.gate_up_proj", f"layer {idx} shared expert"))
    return [(prefix, why) for prefix, why in prefixes if prefix.split(".")[-1] in mapping]


def candidate_prefixes_from_desc(desc, mapping):
    """Fused prefixes implied by the description's own keys.

    Safety net for naming the config-driven walk does not predict.  Candidates
    sitting inside an individual expert (".experts.<i>.gate_proj") are dropped:
    the model never builds a gate_up_proj there, the "experts" group covers it.
    """
    inner_expert = re.compile(r"\.experts\.\d+\.")
    candidates = {}
    for packed_name, shards in mapping.items():
        for shard in shards:
            suffix = f".{shard}.weight"
            for key in desc:
                if not key.endswith(suffix):
                    continue
                base = key[: -len(f"{shard}.weight")]
                prefix = base + packed_name
                if packed_name != "experts" and inner_expert.search(base):
                    continue
                candidates.setdefault(prefix, f"description key *.{shard}.weight")
    return candidates


def classify(desc, prefix, mapping):
    try:
        quant_type = get_quant_type_for_layer(desc, prefix, mapping)
    except KeyError as exc:
        return "MISSING", f"no description entry for {exc.args[0]}"
    except ValueError as exc:
        return "MIXED", str(exc).split(". Please check")[0]
    if quant_type is None:
        return "UNQUANT", "all shards FLOAT or absent -> unquantized method"
    return "OK", quant_type


def report_section(title, results, failures):
    print(f"--- {title} ---")
    if not results:
        print("  nothing to check")
        print()
        return
    buckets = defaultdict(list)
    for prefix, why, verdict, detail in results:
        buckets[verdict].append((prefix, why, detail))
    for verdict in ("MISSING", "MIXED", "UNQUANT", "OK"):
        rows = buckets.get(verdict)
        if not rows:
            continue
        print(f"  {verdict}: {len(rows)}")
        if verdict == "OK":
            types = Counter(detail for _, _, detail in rows)
            for quant_type, count in types.most_common():
                print(f"    {quant_type}: {count} fused module(s)")
            continue
        if verdict == "UNQUANT":
            groups = Counter(prefix.split(".")[-1] for prefix, _, _ in rows)
            for group, count in groups.most_common():
                print(f"    {group}: {count}")
            for prefix, why, detail in rows[:3]:
                print(f"    e.g. {prefix}  ({why})")
            continue
        for prefix, why, detail in rows[:MAX_EXAMPLES]:
            print(f"    {prefix}")
            print(f"      {why}: {detail}")
        if len(rows) > MAX_EXAMPLES:
            print(f"    ... {len(rows) - MAX_EXAMPLES} more")
    for verdict in ("MISSING", "MIXED"):
        rows = buckets.get(verdict)
        if rows:
            failures.append(f"{title}: {len(rows)} fused module(s) would raise {verdict}")
    print()
    return buckets


def report_pre_pr_delta(desc, rows, mapping):
    """Say whether #16051 is what breaks these prefixes, or they were broken already."""
    changed = []
    for prefix, _why, verdict, _detail in rows:
        if verdict not in ("MISSING", "MIXED"):
            continue
        old_verdict, old_detail = _pre_pr_verdict(desc, prefix, mapping)
        if old_verdict == "ok":
            changed.append((prefix, old_detail))
    print("--- is this a #16051 regression? ---")
    failing = [row for row in rows if row[2] in ("MISSING", "MIXED")]
    if not failing:
        print("  nothing to compare: no fused module fails under the post-PR logic")
    elif not changed:
        print(f"  no: all {len(failing)} failing fused module(s) fail under the pre-PR logic too")
        print("  (so a vendor rebase is not what introduces it -- the checkpoint is the problem)")
    else:
        print(f"  yes for {len(changed)} fused module(s): pre-PR they passed, post-PR they raise")
        for prefix, old_detail in changed[:MAX_EXAMPLES]:
            print(f"    {prefix}  (pre-PR: {old_detail})")
        if len(changed) > MAX_EXAMPLES:
            print(f"    ... {len(changed) - MAX_EXAMPLES} more")
    print()


def report_sibling_scan(desc, mapping, warnings):
    """Mapping-independent sweep: sibling .weight entries that disagree.

    Catches a mixed fused group whose packed name the table above does not know
    about, which is exactly what a future vLLM mapping change would introduce.
    """
    print("--- mapping-independent sibling scan ---")
    parents = defaultdict(dict)
    for key, value in desc.items():
        if not key.endswith(".weight") or not isinstance(value, str):
            continue
        module = key[: -len(".weight")]
        if "." not in module:
            continue
        parent, _, leaf = module.rpartition(".")
        parents[parent][leaf] = value
    disagreeing = {parent: leaves for parent, leaves in parents.items() if len(set(leaves.values())) > 1}
    print(f"  {len(parents)} parent module(s) hold at least one described .weight entry")
    print(f"  {len(disagreeing)} of them hold children whose quant types differ")
    if not disagreeing:
        print()
        return
    shapes = Counter()
    for leaves in disagreeing.values():
        signature = tuple(sorted(f"{leaf}={quant}" for leaf, quant in leaves.items()))
        shapes[signature] += 1
    for signature, count in shapes.most_common(MAX_EXAMPLES):
        print(f"    x{count}: " + ", ".join(signature))
    if len(shapes) > MAX_EXAMPLES:
        print(f"    ... {len(shapes) - MAX_EXAMPLES} more shapes")
    covered = set()
    for parent, leaves in disagreeing.items():
        for packed_name, shards in mapping.items():
            if any(leaf in shards for leaf in leaves):
                covered.add(f"{parent}.{packed_name}")
    if covered:
        print(f"  {len(covered)} fused module(s) built from those children are read by the check")
    else:
        warnings.append(
            "sibling disagreement exists but no current fused group covers it; "
            "a vLLM mapping change could turn it into a hard failure"
        )
    print()


def report_expert_spread(desc, warnings):
    """Experts 1..N-1 are never compared: the mapping only names experts.0.*."""
    print("--- expert shards beyond expert 0 (outside the check) ---")
    pattern = re.compile(r"^(?P<block>.*\.experts)\.(?P<idx>\d+)\.(?P<shard>\w+)\.weight$")
    per_block = defaultdict(lambda: defaultdict(dict))
    for key, value in desc.items():
        match = pattern.match(key)
        if match and isinstance(value, str):
            per_block[match.group("block")][match.group("shard")][int(match.group("idx"))] = value
    if not per_block:
        print("  no per-expert weight entries in the description")
        print()
        return
    total_blocks = len(per_block)
    offenders = []
    for block, shards in sorted(per_block.items()):
        for shard, by_idx in sorted(shards.items()):
            types = set(by_idx.values())
            if len(types) > 1:
                expert0 = by_idx.get(0)
                others = sorted(types - {expert0})
                offenders.append((block, shard, expert0, others, len(by_idx)))
    print(f"  {total_blocks} expert block(s) described")
    if not offenders:
        print("  every expert in a block shares its shard's quant type, so expert 0 is representative")
        print()
        return
    print(f"  {len(offenders)} (block, shard) pair(s) disagree across experts")
    for block, shard, expert0, others, count in offenders[:MAX_EXAMPLES]:
        print(f"    {block}.*.{shard}  expert0={expert0}  others={others}  (n={count})")
    if len(offenders) > MAX_EXAMPLES:
        print(f"    ... {len(offenders) - MAX_EXAMPLES} more")
    warnings.append(
        f"{len(offenders)} expert shard(s) vary across experts; the fused check only reads "
        "experts.0, so this passes init and has to break during weight loading instead"
    )
    print()


def report_mtp(desc, text_config, namespace, warnings):
    """The drafter's mapping has no GDN pair, so its in_proj lookups go unpacked."""
    mtp_layers = text_config.get("mtp_num_hidden_layers") or 0
    if not mtp_layers:
        return
    print("--- MTP draft layers ---")
    num_layers = text_config.get("num_hidden_layers") or 0
    mtp_mapping = BUILTIN_PACKED_MODULES_MAPPING["Qwen3_5MoeMTP"]
    found_gdn = []
    for idx in range(num_layers, num_layers + mtp_layers):
        base = f"{namespace}layers.{idx}."
        described = [key for key in desc if key.startswith(base)]
        print(f"  layer {idx}: {len(described)} description entry/entries")
        if not described:
            warnings.append(
                f"MTP layer {idx} has no quant description entries; the drafter's linear layers "
                "resolve to None post-PR and load unquantized instead of raising"
            )
            continue
        for shard in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
            if any(key.endswith(f".{shard}.weight") for key in described):
                found_gdn.append((idx, shard))
    if found_gdn:
        shards = sorted({shard for _, shard in found_gdn})
        print(f"  GDN shards described on MTP layers: {shards}")
        print("  Qwen3_5MoeMTP.packed_modules_mapping has no in_proj_qkvz/in_proj_ba entry,")
        print("  so these resolve through the non-packed branch and come out unquantized.")
        warnings.append(
            "MTP GDN shards are described but the drafter mapping does not pack them; "
            "the draft GDN projection loads unquantized"
        )
    print(f"  drafter fused groups that are checked: {sorted(mtp_mapping)}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_path", nargs="?", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--desc", help="quant_model_description.json (defaults to MODEL_PATH/...)")
    parser.add_argument("--config", help="config.json (defaults to MODEL_PATH/config.json)")
    parser.add_argument(
        "--from-vllm",
        action="store_true",
        help="import the real vLLM model class for packed_modules_mapping instead of the built-in table",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    desc_path = Path(args.desc) if args.desc else model_path / "quant_model_description.json"
    config_path = Path(args.config) if args.config else model_path / "config.json"

    failures = []
    warnings = []

    print("=== inputs ===")
    print(f"  model path  : {model_path}")
    print(f"  description : {desc_path}")
    print(f"  config      : {config_path}")
    raw_desc = load_json(desc_path)
    config = load_json(config_path)
    if raw_desc is None or config is None:
        print()
        print("  [FAIL] need both config.json and quant_model_description.json")
        print()
        print("[RED]")
        return 1
    print()

    architecture = (config.get("architectures") or ["?"])[0]
    model_type = config.get("model_type") or "?"
    text_config = config.get("text_config") or config
    print("=== model ===")
    print(f"  architecture : {architecture}")
    print(f"  model_type   : {model_type}   (this is what get_quant_method() keys on)")
    print(f"  text type    : {text_config.get('model_type')}")
    print(f"  layers       : {text_config.get('num_hidden_layers')}, mtp {text_config.get('mtp_num_hidden_layers')}")
    print(
        f"  experts      : {text_config.get('num_experts')}, shared {text_config.get('shared_expert_intermediate_size')}"
    )
    if model_type in UPDATED_PACKED_MODULES_MAPPING:
        print(f"  {model_type} IS in UPDATED_PACKED_MODULES_MAPPING (vllm-ascend still overrides it)")
    else:
        print(f"  {model_type} is NOT in UPDATED_PACKED_MODULES_MAPPING -> mapping comes from vLLM alone")
    print()

    desc = apply_extra_quant_adaptations({k: v for k, v in raw_desc.items() if k != "optional"})
    weight_entries = {k: v for k, v in desc.items() if k.endswith(".weight") and isinstance(v, str)}
    print("=== quant description ===")
    print(f"  {len(raw_desc)} raw key(s), {len(desc)} after ModelSlim aliasing, {len(weight_entries)} *.weight entries")
    for key in ("model_quant_type", "kv_cache_type", "fa_quant_type", "indexer_quant_type", "group_size"):
        if key in raw_desc:
            print(f"  {key} = {raw_desc[key]!r}")
    histogram = Counter(weight_entries.values())
    print("  quant type histogram over *.weight entries:")
    for quant_type, count in histogram.most_common():
        print(f"    {quant_type}: {count}")
    if len(histogram) == 1:
        print("  single quant type across the whole checkpoint -> no fused group can be mixed")
    print()

    mapping, source = resolve_packed_mapping(architecture, model_type, desc, args.from_vllm)
    if mapping is None:
        print("  [FAIL] cannot resolve packed_modules_mapping")
        print()
        print("[RED]")
        return 1
    print("=== packed_modules_mapping in effect ===")
    print(f"  source: {source}")
    for packed_name in sorted(mapping):
        print(f"    {packed_name} <- {mapping[packed_name]}")
    print()

    namespace = detect_layer_namespace(desc)
    if namespace is None:
        print("  no 'layers.<i>.' keys in the description; falling back to the description sweep only")
        namespace = ""
        config_rows = []
    else:
        print(f"=== detected layer namespace: {namespace!r} ===")
        print()
        config_rows = [
            (prefix, why) + classify(desc, prefix, mapping)
            for prefix, why in expected_prefixes_from_config(text_config, namespace, mapping)
        ]

    config_prefixes = {row[0] for row in config_rows}
    desc_rows = [
        (prefix, why) + classify(desc, prefix, mapping)
        for prefix, why in sorted(candidate_prefixes_from_desc(desc, mapping).items())
        if prefix not in config_prefixes
    ]

    print("=== replayed get_quant_type_for_layer ===")
    report_section("fused modules the model builds (from config.json)", config_rows, failures)
    report_section("further fused modules implied by description keys", desc_rows, failures)
    report_pre_pr_delta(desc, config_rows + desc_rows, mapping)

    print("=== outside the replayed check ===")
    report_sibling_scan(desc, mapping, warnings)
    report_expert_spread(desc, warnings)
    if namespace:
        report_mtp(desc, text_config, namespace, warnings)

    print("=== verdict ===")
    for warning in warnings:
        print("  [WARN] " + warning)
    for failure in failures:
        print("  [FAIL] " + failure)
    if failures:
        print()
        print("  This checkpoint trips PR #16051's fused-shard check: model init raises before")
        print("  any weight is loaded.  Fix the description (make every shard of a fused module")
        print("  carry one quant type, and describe every shard the model builds) or re-export.")
        print()
        print("[RED]")
        return 1
    print()
    print("  Every fused module the model builds has all its shards described with one quant")
    print("  type, so the check passes and model init gets past quantization setup.")
    print("[GREEN]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
