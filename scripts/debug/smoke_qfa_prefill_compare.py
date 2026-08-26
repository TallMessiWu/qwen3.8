#!/usr/bin/env python3
"""Compare QFA MXFP8 prefill against the BF16 FIA baseline on the 4-layer model.

Runs the same greedy generation twice in subprocesses — once with
VLLM_ASCEND_ENABLE_QFA_PREFILL=0 (BF16 FIA baseline) and once with =1 (MXFP8
QuantFlashAttn prefill) — on the four-layer single-node Qwen3.8 smoke
configuration (3x linear_attention + 1x full_attention), then compares the
first-token top-5 logprobs and the greedy token prefix.

GREEN criteria: both runs finish, first greedy token matches, and the
first-token top-5 candidate sets overlap by >= 4/5. Token-prefix divergence
beyond the first token is reported but does not gate (MXFP8 rounding may
legitimately flip near-ties on a sliced 4-layer model).

Environment:
  MODEL_PATH     default /mnt/share/weight/Qwen3.8-2.4T-A95B-mxfp8
  FOUR_LAYER     1 (default): apply the 4-layer hf_overrides + checkpoint layer
                 filter + expert parallel (the 2.4T smoke setup);
                 0: load the model as-is (e.g. Qwen3.8-27B)
  QUANTIZATION   ascend (default) passes --quantization ascend; anything else
                 (auto/none) lets vLLM auto-detect the checkpoint quant config
  TP_SIZE        default 8

27B single-card example (mirrors scripts/27B.sh):
  MODEL_PATH=/mnt/share/weight/Qwen3.8-27B-mxfp8 FOUR_LAYER=0 QUANTIZATION=auto \
      TP_SIZE=1 python scripts/debug/smoke_qfa_prefill_compare.py

The QFA run must print "QFA MXFP8 prefill path engaged" (from vllm-ascend);
if it does not, the model never hit the quantized branch and the comparison is
vacuous.
"""

import json
import os
import subprocess
import sys
import tempfile

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/share/weight/Qwen3.8-2.4T-A95B-mxfp8")
QUANTIZATION = os.environ.get("QUANTIZATION", "ascend")
TP_SIZE = int(os.environ.get("TP_SIZE", "8"))
FOUR_LAYER = os.environ.get("FOUR_LAYER", "1") == "1"
PROMPTS = [
    "The capital of France is",
    "请用一句话介绍一下人工智能。",
]
MAX_TOKENS = 32
TOP_LOGPROBS = 5


def child(result_path: str) -> None:
    from vllm import LLM, SamplingParams

    kwargs = {}
    if FOUR_LAYER:
        kwargs["hf_overrides"] = {
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        }
        kwargs["enable_expert_parallel"] = True
    if QUANTIZATION == "ascend":
        kwargs["quantization"] = "ascend"
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=TP_SIZE,
        max_model_len=2048,
        max_num_seqs=1,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        seed=1024,
        safetensors_load_strategy="lazy",
        **kwargs,
    )
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=TOP_LOGPROBS)
    outputs = llm.generate(PROMPTS, params)
    result = []
    for out in outputs:
        seq = out.outputs[0]
        first = seq.logprobs[0] if seq.logprobs else {}
        result.append(
            {
                "token_ids": list(seq.token_ids),
                "text": seq.text,
                "first_top": {str(tid): lp.logprob for tid, lp in first.items()},
            }
        )
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


def run_child(qfa: int, result_path: str) -> None:
    env = dict(os.environ)
    env["VLLM_ASCEND_ENABLE_QFA_PREFILL"] = str(qfa)
    layer_filter = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "runtime", "qwen38_checkpoint_layer_filter")
    if FOUR_LAYER and os.path.isdir(layer_filter):
        env["QWEN38_CHECKPOINT_LAYER_LIMIT"] = "4"
        env["PYTHONPATH"] = layer_filter + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("VLLM_ASCEND_ENABLE_FUSED_MC2", "0")
    print(f"[INFO] launching child VLLM_ASCEND_ENABLE_QFA_PREFILL={qfa} ...", flush=True)
    subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", result_path],
        env=env,
        check=True,
    )


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
        return 0

    if not os.path.isfile(os.path.join(MODEL_PATH, "config.json")):
        print(f"[RED] missing {MODEL_PATH}/config.json")
        return 2

    with tempfile.TemporaryDirectory(prefix="qfa_smoke_") as tmp:
        base_path = os.path.join(tmp, "base.json")
        qfa_path = os.path.join(tmp, "qfa.json")
        run_child(0, base_path)
        run_child(1, qfa_path)
        with open(base_path, encoding="utf-8") as f:
            base = json.load(f)
        with open(qfa_path, encoding="utf-8") as f:
            qfa = json.load(f)

    all_ok = True
    for i, (b, q) in enumerate(zip(base, qfa)):
        b_ids, q_ids = b["token_ids"], q["token_ids"]
        prefix = 0
        for x, y in zip(b_ids, q_ids):
            if x != y:
                break
            prefix += 1
        b_top = set(b["first_top"])
        q_top = set(q["first_top"])
        overlap = len(b_top & q_top)
        first_same = bool(b_ids and q_ids and b_ids[0] == q_ids[0])
        print(f"== prompt {i}: {PROMPTS[i]!r}")
        print(f"  baseline: {b['text']!r}")
        print(f"  qfa:      {q['text']!r}")
        print(f"  first_token_same={first_same} top{TOP_LOGPROBS}_overlap={overlap}/{TOP_LOGPROBS} "
              f"greedy_prefix_match={prefix}/{min(len(b_ids), len(q_ids))}")
        for tid in sorted(b_top & q_top):
            delta = abs(b["first_top"][tid] - q["first_top"][tid])
            print(f"    token {tid}: logprob base={b['first_top'][tid]:.4f} qfa={q['first_top'][tid]:.4f} "
                  f"delta={delta:.4f}")
        ok = first_same and overlap >= TOP_LOGPROBS - 1
        print(f"  [{'GREEN' if ok else 'RED'}] prompt {i}")
        all_ok = all_ok and ok

    print(f"[{'GREEN' if all_ok else 'RED'}] QFA prefill vs BF16 baseline overall")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
