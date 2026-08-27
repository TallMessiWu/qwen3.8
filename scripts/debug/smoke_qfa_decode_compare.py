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
  PROMPT_IDX   run only ALL_PROMPTS[idx] (isolates batching from content)

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
import time

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/share/weight/Qwen3.8-27B-mxfp8")
TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
NUM_SPEC = int(os.environ.get("NUM_SPEC", "3"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
PREFIX_GATE = int(os.environ.get("PREFIX_GATE", "8"))
ALL_PROMPTS = [
    "The capital of France is",
    "请用一句话介绍一下人工智能。",
    "Count from one to twenty in English words:",
]
# PROMPT_IDX isolates one prompt so a batched-only failure can be told apart
# from a content-dependent one.
_IDX = os.environ.get("PROMPT_IDX")
PROMPTS = [ALL_PROMPTS[int(_IDX)]] if _IDX else ALL_PROMPTS
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
        max_model_len=int(os.environ.get("MAX_LEN", "4096")),
        max_num_seqs=4,
        # QFA decode allocates its own MXFP8 planes on top of the BF16 cache,
        # so leave it headroom: both shrink together as this comes down.
        gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.85")),
        enforce_eager=True,
        seed=1024,
        safetensors_load_strategy="lazy",
        **kwargs,
    )
    tok = llm.get_tokenizer()
    lens = [len(tok.encode(p)) for p in PROMPTS]
    print(f"[child] engine ready, prompts={len(PROMPTS)} token_lens={lens} x {MAX_TOKENS} tokens", flush=True)
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=TOP_LOGPROBS)
    outputs = llm.generate(PROMPTS, params)
    print("[child] generation done", flush=True)
    result = []
    for out in outputs:
        seq = out.outputs[0]
        first = seq.logprobs[0] if seq.logprobs else {}
        result.append(
            {
                "token_ids": list(seq.token_ids),
                "text": seq.text,
                "first_top": {str(tid): lp.logprob for tid, lp in first.items()},
                # Every step's top-k, not just the first. Greedy can diverge
                # well past token 0, and only the steps inside the shared
                # prefix are comparable at all - there both runs fed the model
                # the same context, so a logprob difference is the quantized
                # KV cache's numeric error and nothing else.
                "steps": [
                    {str(tid): [lp.logprob, lp.decoded_token] for tid, lp in step.items()}
                    for step in (seq.logprobs or [])
                ],
            }
        )
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


# A 27B load takes minutes; without live output the script looks hung. These
# patterns are surfaced as they arrive, everything else is kept for the report.
LIVE_RE = re.compile(
    r"QFA MXFP8|QFA-DEBUG|QFA-CACHE|Traceback|Error|ERROR|Exception|AssertionError|"
    r"Loading|Memory profiling|KV cache|Capturing|Adding requests|Processed prompts|"
    # Page sizing decides whether the MXFP8 cache actually saves memory: the
    # mamba page is padded to a BF16 attention page during platform setup, so
    # these lines are what say the halved page survived.
    r"attention block size|Padding mamba|[Cc]oncurrency|"
    r"[Aa]cceptance|it/s|\[child\]"
)


def run_child(qfa: int, result_path: str) -> str:
    env = dict(os.environ)
    env["VLLM_ASCEND_ENABLE_QFA_PREFILL"] = str(qfa)
    env["VLLM_ASCEND_ENABLE_QFA_DECODE"] = str(qfa)
    env.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("VLLM_ASCEND_ENABLE_FUSED_MC2", "0")
    env.setdefault("PYTHONUNBUFFERED", "1")
    tag = "qfa " if qfa else "base"
    print(
        f"[INFO] launching child {tag.strip()} "
        f"(VLLM_ASCEND_ENABLE_QFA_PREFILL/DECODE={qfa}); filtered child output follows",
        flush=True,
    )
    started = time.time()
    proc = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--child", result_path],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    lines: list[str] = []
    last_print = started
    for line in proc.stdout:
        lines.append(line)
        now = time.time()
        if LIVE_RE.search(line):
            print(f"  [{tag} {now - started:5.0f}s] {line.rstrip()}", flush=True)
            last_print = now
        elif now - last_print >= 30:
            # Heartbeat so a long quiet stretch is not mistaken for a hang.
            print(f"  [{tag} {now - started:5.0f}s] ... alive, {len(lines)} log lines", flush=True)
            last_print = now
    proc.wait()
    log = "".join(lines)
    print(f"[INFO] child {tag.strip()} finished in {time.time() - started:.0f}s (rc={proc.returncode})", flush=True)
    if proc.returncode != 0:
        print(f"[RED] child {tag.strip()} exited {proc.returncode}; last 40 lines:")
        for tail_line in log.splitlines()[-40:]:
            print(f"    {tail_line}")
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


def _logprob(step: dict, tid: str) -> str:
    entry = step.get(tid)
    return f"{entry[0]:8.4f}" if entry else "  <off-topk>"


def _top_gap(step: dict) -> float:
    """How close this step was to flipping, ignoring the other run entirely."""
    ranked = sorted((entry[0] for entry in step.values()), reverse=True)
    return ranked[0] - ranked[1] if len(ranked) > 1 else float("inf")


def report_numerics(b: dict, q: dict, prefix: int) -> None:
    """Separate "the KV cache is wrong" from "greedy tipped over at a coin flip".

    Two numbers decide it. Over the shared prefix both runs saw identical
    context, so the delta on the token they both chose is a direct read of the
    quantized cache's error - a few thousandths is quantization, tenths is a
    bug. At the divergent step, the top1/top2 gap says how much error it would
    have taken to flip: a gap below the measured delta means the flip is
    explained by that error and nothing else has to be wrong.
    """
    b_steps, q_steps = b.get("steps") or [], q.get("steps") or []
    if not b_steps or not q_steps:
        print("  [WARN] no per-step logprobs in the result; rerun with the current script")
        return
    deltas = []
    for i in range(min(prefix, len(b_steps), len(q_steps))):
        tid = str(b["token_ids"][i])
        if tid in b_steps[i] and tid in q_steps[i]:
            deltas.append(abs(b_steps[i][tid][0] - q_steps[i][tid][0]))
    if deltas:
        print(
            f"  chosen-token logprob delta across the shared prefix: "
            f"max={max(deltas):.4f} mean={sum(deltas) / len(deltas):.4f} n={len(deltas)}"
        )
    if prefix >= min(len(b_steps), len(q_steps)):
        print("  no divergence inside the logged steps")
        return
    b_step, q_step = b_steps[prefix], q_steps[prefix]
    print(f"  divergence at step {prefix}:")
    for who, tid in (("baseline", str(b["token_ids"][prefix])), ("qfa     ", str(q["token_ids"][prefix]))):
        label = (b_step.get(tid) or q_step.get(tid) or [0.0, "?"])[1]
        print(
            f"    {who} picked {tid:>7} {label!r:<14} "
            f"base={_logprob(b_step, tid)} qfa={_logprob(q_step, tid)}"
        )
    print(f"    top1-top2 gap: base={_top_gap(b_step):.4f} qfa={_top_gap(q_step):.4f}")


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
        # QFA first: it is the run that can fail, and waiting out a full
        # baseline before finding that out wastes minutes every iteration.
        qfa_log = run_child(1, qfa_path)
        if os.environ.get("QFA_ONLY") == "1":
            # Instrumentation runs only need the QFA child; the baseline costs
            # another three minutes and says nothing about the debug output.
            with open(qfa_path, encoding="utf-8") as f:
                qfa = json.load(f)
            report_log("qfa", qfa_log)
            for i, q in enumerate(qfa):
                print(f"== prompt {i}: {PROMPTS[i]!r}")
                print(f"  qfa: {q['text']!r}")
            print("[INFO] QFA_ONLY=1: skipped the baseline, no verdict")
            return 0
        base_log = run_child(0, base_path)
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
        report_numerics(b, q, prefix)
        ok = first_same and overlap >= TOP_LOGPROBS - 1 and prefix >= min(PREFIX_GATE, shared)
        print(f"  [{'GREEN' if ok else 'RED'}] prompt {i}")
        all_ok = all_ok and ok

    print(f"[{'GREEN' if all_ok else 'RED'}] QFA MXFP8 decode vs BF16 baseline overall")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
