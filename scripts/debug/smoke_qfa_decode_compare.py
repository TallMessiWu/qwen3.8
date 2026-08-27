#!/usr/bin/env python3
"""Compare the MXFP8 QFA decode path against the BF16 baseline end to end.

Milestone C moves the full-attention KV cache to MXFP8 and serves decode, MTP
verify and chunked prefill from the paged QuantFlashAttn op. This runs the same
greedy generation twice in subprocesses - once with both QFA switches off (BF16
FIA everywhere) and once with both on - with MTP speculative decoding enabled so
the verify path is exercised, then compares outputs and acceptance behaviour.

GREEN criteria: both runs finish, the QFA run reports the paged path engaged,
the first greedy token matches per prompt, and the greedy prefixes agree for at
least PREFIX_GATE tokens. Quantized KV shifts logits slightly, so a late
divergence is reported rather than gated; the acceptance rate is printed
because a large drop there is the signal that verify accuracy suffered.

Environment:
  MODEL_PATH   default /mnt/share/weight/Qwen3.8-27B-mxfp8
  TP_SIZE      default 1
  NUM_SPEC     default 3 (0 disables MTP, exercising plain DecodeOnly instead)
  MAX_TOKENS   default 64
  PREFIX_GATE  default 8

Example (27B single card, mirrors scripts/27B.sh):
  MODEL_PATH=/mnt/share/weight/Qwen3.8-27B-mxfp8 TP_SIZE=1 \
      python scripts/debug/smoke_qfa_decode_compare.py

Requires enforce_eager: aclgraph capture of the paged path lands in C3.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/share/weight/Qwen3.8-27B-mxfp8")
TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
NUM_SPEC = int(os.environ.get("NUM_SPEC", "3"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
PREFIX_GATE = int(os.environ.get("PREFIX_GATE", "8"))
PROMPTS = [
    "The capital of France is",
    "请用一句话介绍一下人工智能。",
    "Count from one to twenty in English words:",
]
TOP_LOGPROBS = 5
ACCEPT_RE = re.compile(r"(?:acceptance length|Acceptance rate|accepted)[^\n]*", re.IGNORECASE)


def child(result_path: str) -> None:
    from vllm import LLM, SamplingParams

    kwargs = {}
    if NUM_SPEC > 0:
        kwargs["speculative_config"] = {
            "method": "qwen3_5_mtp",
            "num_speculative_tokens": NUM_SPEC,
        }
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=TP_SIZE,
        max_model_len=4096,
        max_num_seqs=4,
        gpu_memory_utilization=0.85,
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


def run_child(qfa: int, result_path: str) -> str:
    env = dict(os.environ)
    env["VLLM_ASCEND_ENABLE_QFA_PREFILL"] = str(qfa)
    env["VLLM_ASCEND_ENABLE_QFA_DECODE"] = str(qfa)
    env.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("VLLM_ASCEND_ENABLE_FUSED_MC2", "0")
    print(f"[INFO] launching child with QFA={qfa} (prefill+decode) ...", flush=True)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", result_path],
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,  # the tail of a failing child is more useful than a traceback
    )
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        tail = "\n".join(log.splitlines()[-40:])
        print(f"[RED] child QFA={qfa} exited {proc.returncode}\n{tail}")
        raise SystemExit(1)
    return log


def report_log(tag: str, log: str) -> dict:
    marks = {
        "prefill_engaged": "QFA MXFP8 prefill path engaged" in log,
        "paged_engaged": "QFA MXFP8 paged path engaged" in log,
    }
    print(f"  [{tag}] prefill_engaged={marks['prefill_engaged']} paged_engaged={marks['paged_engaged']}")
    accept = [line.strip() for line in log.splitlines() if ACCEPT_RE.search(line)]
    for line in accept[-3:]:
        print(f"  [{tag}] {line}")
    marks["accept_lines"] = accept[-3:]
    return marks


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
        return 0

    if not os.path.isfile(os.path.join(MODEL_PATH, "config.json")):
        print(f"[RED] missing {MODEL_PATH}/config.json")
        return 2

    with tempfile.TemporaryDirectory(prefix="qfa_dec_") as tmp:
        base_path = os.path.join(tmp, "base.json")
        qfa_path = os.path.join(tmp, "qfa.json")
        base_log = run_child(0, base_path)
        qfa_log = run_child(1, qfa_path)
        with open(base_path, encoding="utf-8") as f:
            base = json.load(f)
        with open(qfa_path, encoding="utf-8") as f:
            qfa = json.load(f)

    print("== run markers ==")
    report_log("baseline", base_log)
    qfa_marks = report_log("qfa", qfa_log)
    if not qfa_marks["paged_engaged"]:
        print("[RED] the QFA run never entered the paged path - the comparison is vacuous")
        return 1

    all_ok = True
    for i, (b, q) in enumerate(zip(base, qfa)):
        b_ids, q_ids = b["token_ids"], q["token_ids"]
        prefix = 0
        for x, y in zip(b_ids, q_ids):
            if x != y:
                break
            prefix += 1
        b_top, q_top = set(b["first_top"]), set(q["first_top"])
        overlap = len(b_top & q_top)
        first_same = bool(b_ids and q_ids and b_ids[0] == q_ids[0])
        shared = min(len(b_ids), len(q_ids))
        print(f"== prompt {i}: {PROMPTS[i]!r}")
        print(f"  baseline: {b['text']!r}")
        print(f"  qfa:      {q['text']!r}")
        print(
            f"  first_token_same={first_same} top{TOP_LOGPROBS}_overlap={overlap}/{TOP_LOGPROBS} "
            f"greedy_prefix_match={prefix}/{shared} (gate {PREFIX_GATE})"
        )
        for tid in sorted(b_top & q_top):
            delta = abs(b["first_top"][tid] - q["first_top"][tid])
            print(
                f"    token {tid}: logprob base={b['first_top'][tid]:.4f} "
                f"qfa={q['first_top'][tid]:.4f} delta={delta:.4f}"
            )
        ok = first_same and overlap >= TOP_LOGPROBS - 1 and prefix >= min(PREFIX_GATE, shared)
        print(f"  [{'GREEN' if ok else 'RED'}] prompt {i}")
        all_ok = all_ok and ok

    print(f"[{'GREEN' if all_ok else 'RED'}] QFA MXFP8 decode vs BF16 baseline overall")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
