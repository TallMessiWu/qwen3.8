#!/usr/bin/env python3
"""Compare the MXFP8 QFA decode path against the BF16 baseline end to end.

Milestone C moves the full-attention KV cache to MXFP8 and serves decode, MTP
verify and chunked prefill from the paged QuantFlashAttn op. This runs the same
greedy generation twice in subprocesses - once with both QFA switches off (BF16
FIA everywhere) and once with both on - with MTP speculative decoding enabled so
the verify path is exercised, then compares outputs and acceptance behaviour.

GREEN criteria: both runs finish, the QFA run reports the paged path engaged,
and on the steps where the two runs saw identical context the quantized cache
moved the logprobs by no more than DELTA_GATE while keeping OVERLAP_GATE of
the top-k. Greedy prefix length is printed but NOT gated: measured on 27B, one
prompt's chains agreed for 47 tokens and another's for 2 with the same
underlying error - what differed was only how soon a near-tie came up, and a
tie flips on a fraction of a bf16 ulp. The acceptance rate is printed because
a large drop there is the signal that verify accuracy suffered.

NUM_SPEC=0 is the real accuracy check. Speculation gets a 3x looser gate for
the reason spelled out at DELTA_GATE, loose enough that a defect shifting
logprobs by 1.2 would pass it, so a green speculative run means the verify
path did not make things much worse - not that the cache is accurate.
check_smoke_gate_offline.py replays measured numbers through both gates and
pins down that gap; run it after touching any of them.

Environment:
  MODEL_PATH    default /mnt/share/weight/Qwen3.8-27B-mxfp8
  TP_SIZE       default 1
  NUM_SPEC      default 3 (0 disables MTP, exercising plain DecodeOnly instead)
  MAX_TOKENS    default 64
  SELF_SPEC     1 compares the QFA path against ITSELF with NUM_SPEC=0 rather
                than against BF16, isolating the verify path from quantization
  DELTA_GATE    default 0.5, or 1.5 when NUM_SPEC>0 - max chosen-token shift
  OVERLAP_GATE  default 0.8 - min mean top-k overlap, as a fraction of k
  TOP_LOGPROBS  default 10
  PROMPT_IDX    run only ALL_PROMPTS[idx] (isolates batching from content)

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
# SELF_SPEC=1 runs the quantized path against ITSELF with speculation off,
# instead of against BF16, which isolates what speculation alone costs.
#
# It does NOT gate on the two agreeing, though the arithmetic argument says it
# should: at temperature 0 rejection sampling only accepts a draft token that
# already is the target's argmax, so the target alone decides the output, and
# BF16 does reproduce its NUM_SPEC=0 output exactly. QFA does not, because its
# V scale is shared across a 64-token window: a verify step appends up to
# 1 + num_spec tokens before the attention read, so that read sees a scale a
# non-speculative run reaches only later, and the same history comes back
# quantized differently. Measured on 27B: outputs identical for 34 steps with
# a 0.16 logprob delta throughout. So the same gate as the BF16 comparison
# applies here - what this mode buys is a delta with the quantization error
# already cancelled out, not an exact-match check.
SELF_SPEC = os.environ.get("SELF_SPEC") == "1"
# The gate is the numeric error, not the length of the agreeing token chain.
# 0.5 sits well clear of the 0.27 worst case measured without speculation,
# while still catching the order-of-magnitude jump a broken cache produces.
#
# Speculation gets its own gate rather than loosening that one, which would
# let real defects through on the path that has none. A verify step writes
# 1 + num_spec tokens before the read, and the V scale it shares across a
# 32-token group moves with every one of them: measured on 27B at num_spec=3,
# a step moves 12% of the scale bytes against 4% without speculation, 2.6x at
# a group boundary rising to 6.3x mid-group. The same underlying error
# therefore surfaces several times larger - worst observed 0.95 - so 1.5
# keeps roughly the same headroom over the measured worst case that 0.5 does.
DELTA_GATE = float(os.environ.get("DELTA_GATE", "1.5" if NUM_SPEC > 0 else "0.5"))
OVERLAP_GATE = float(os.environ.get("OVERLAP_GATE", "0.8"))
# Top-k membership churns on its own wherever the top two candidates sit close
# together: the ranks below them are settled by hundredths of a logprob, and
# quantization noise reshuffles those without ever touching the winner. So the
# overlap is only counted on steps whose top-1 is clearly ahead, where a
# reshuffle means something. Measured on 27B: a run with all 64 chosen tokens
# identical to BF16 still averaged 8.08/10 overlap - 0.08 off failing - purely
# from the flat tail of a list-continuation prompt. Ungated, this statistic
# reports how flat the prompt is, not whether the distribution moved.
OVERLAP_MIN_GAP = float(os.environ.get("OVERLAP_MIN_GAP", "0.5"))
# Below this many comparable steps the verdict rests on too little evidence to
# mean much; it is reported rather than failed, since an early tie is normal.
MIN_STEPS = int(os.environ.get("MIN_STEPS", "3"))
ALL_PROMPTS = [
    "The capital of France is",
    "请用一句话介绍一下人工智能。",
    "Count from one to twenty in English words:",
]
# PROMPT_IDX isolates one prompt so a batched-only failure can be told apart
# from a content-dependent one.
_IDX = os.environ.get("PROMPT_IDX")
PROMPTS = [ALL_PROMPTS[int(_IDX)]] if _IDX else ALL_PROMPTS
# A wider window than the old 5 makes the overlap average steady and gives the
# baseline's pick room to slip a few ranks without falling off the edge.
TOP_LOGPROBS = int(os.environ.get("TOP_LOGPROBS", "10"))
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
        # The FP8 values are the KV cache itself now; only the E8M0 scale side
        # tables sit on top of it, at 1/32 of the values.
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
        result.append(
            {
                "token_ids": list(seq.token_ids),
                "text": seq.text,
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
    r"QFA MXFP8|QFA-DEBUG|QFA-CACHE|QFA-SCALE|Traceback|Error|ERROR|Exception|AssertionError|"
    r"Loading|Memory profiling|KV cache|Capturing|Adding requests|Processed prompts|"
    # Page sizing decides whether the MXFP8 cache actually saves memory: the
    # mamba page is padded to a BF16 attention page during platform setup, so
    # these lines are what say the halved page survived.
    r"attention block size|Padding mamba|[Cc]oncurrency|"
    r"[Aa]cceptance|it/s|\[child\]"
)


def run_child(qfa: int, result_path: str, num_spec: int | None = None, tag: str | None = None) -> str:
    env = dict(os.environ)
    env["VLLM_ASCEND_ENABLE_QFA_PREFILL"] = str(qfa)
    env["VLLM_ASCEND_ENABLE_QFA_DECODE"] = str(qfa)
    if num_spec is not None:
        env["NUM_SPEC"] = str(num_spec)
    env.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("VLLM_ASCEND_ENABLE_FUSED_MC2", "0")
    env.setdefault("PYTHONUNBUFFERED", "1")
    tag = (tag or ("qfa" if qfa else "base")).ljust(4)
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


def report_numerics(b: dict, q: dict, prefix: int, ref_label: str = "baseline", test_label: str = "qfa") -> dict:
    """Measure the cache's error on the steps where the runs are comparable.

    Both runs fed the model identical context for every step up to and
    including the first divergent one, so across that range a logprob
    difference is the MXFP8 cache's error and nothing else. Past it the
    contexts differ and nothing is comparable - which is exactly why the
    length of the agreeing token chain is evidence about tie spacing, not
    about accuracy.

    Returns the numbers main() gates on, and prints the divergent step, where
    the top1/top2 gap says how much error the flip actually required.
    """
    b_steps, q_steps = b.get("steps") or [], q.get("steps") or []
    stats = {
        "steps": 0,
        "max_delta": float("inf"),
        "mean_delta": float("nan"),
        "mean_overlap": 0.0,
        "min_overlap": 0,
        "overlap_steps": 0,
        "off_topk": 0,
    }
    if not b_steps or not q_steps:
        print("  [WARN] no per-step logprobs in the result; rerun with the current script")
        return stats
    # The divergent step is included: its context is still shared, and it is
    # the single most informative step in the run.
    last = min(prefix + 1, len(b_steps), len(q_steps))
    deltas, overlaps, off_topk = [], [], 0
    for i in range(last):
        tid = str(b["token_ids"][i])
        b_lp, q_lp = b_steps[i].get(tid), q_steps[i].get(tid)
        if b_lp and q_lp:
            deltas.append(abs(b_lp[0] - q_lp[0]))
        else:
            # The baseline's pick left the QFA run's top-k entirely, so the
            # shift is larger than this window can measure. Fail loudly
            # instead of quietly dropping the step from the average.
            off_topk += 1
        if _top_gap(b_steps[i]) >= OVERLAP_MIN_GAP:
            overlaps.append(len(set(b_steps[i]) & set(q_steps[i])))
    stats.update(
        steps=last,
        max_delta=float("inf") if off_topk else (max(deltas) if deltas else float("inf")),
        mean_delta=sum(deltas) / len(deltas) if deltas else float("nan"),
        # With no decisive step there is nothing for this statistic to say, so
        # it abstains rather than failing the prompt on the delta's behalf.
        mean_overlap=sum(overlaps) / len(overlaps) if overlaps else float(TOP_LOGPROBS),
        min_overlap=min(overlaps) if overlaps else TOP_LOGPROBS,
        overlap_steps=len(overlaps),
        off_topk=off_topk,
    )
    print(
        f"  comparable steps={stats['steps']} | chosen-token logprob delta "
        f"max={stats['max_delta']:.4f} mean={stats['mean_delta']:.4f} | "
        f"top{TOP_LOGPROBS} overlap mean={stats['mean_overlap']:.2f} min={stats['min_overlap']} "
        f"over {stats['overlap_steps']} decisive step(s)"
        + (f" | {off_topk} step(s) where the baseline's pick left the QFA top-k" if off_topk else "")
    )
    if stats["steps"] < MIN_STEPS:
        print(f"  [WARN] only {stats['steps']} comparable step(s) - an early tie makes this verdict weak")
    if prefix >= min(len(b_steps), len(q_steps)):
        return stats
    b_step, q_step = b_steps[prefix], q_steps[prefix]
    width = max(len(ref_label), len(test_label))
    print(f"  divergence at step {prefix}:")
    for who, tid in ((ref_label, str(b["token_ids"][prefix])), (test_label, str(q["token_ids"][prefix]))):
        label = (b_step.get(tid) or q_step.get(tid) or [0.0, "?"])[1]
        print(
            f"    {who.ljust(width)} picked {tid:>7} {label!r:<14} "
            f"{ref_label}={_logprob(b_step, tid)} {test_label}={_logprob(q_step, tid)}"
        )
    print(f"    top1-top2 gap: base={_top_gap(b_step):.4f} qfa={_top_gap(q_step):.4f}")
    return stats


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
        return 0

    if not os.path.isfile(os.path.join(MODEL_PATH, "config.json")):
        print(f"[RED] missing {MODEL_PATH}/config.json")
        return 2

    test_label, ref_label = (f"spec{NUM_SPEC}", "spec0") if SELF_SPEC else ("qfa", "baseline")
    with tempfile.TemporaryDirectory(prefix="qfa_dec_") as tmp:
        ref_path = os.path.join(tmp, "ref.json")
        test_path = os.path.join(tmp, "test.json")
        # The test run goes first: it is the one that can fail, and waiting out
        # a full reference before finding that out wastes minutes every
        # iteration.
        test_log = run_child(1, test_path, tag=test_label)
        if not SELF_SPEC and os.environ.get("QFA_ONLY") == "1":
            # Instrumentation runs only need the QFA child; the baseline costs
            # another three minutes and says nothing about the debug output.
            with open(test_path, encoding="utf-8") as f:
                test = json.load(f)
            report_log(test_label, test_log)
            for i, t in enumerate(test):
                print(f"== prompt {i}: {PROMPTS[i]!r}")
                print(f"  {test_label}: {t['text']!r}")
            print("[INFO] QFA_ONLY=1: skipped the baseline, no verdict")
            return 0
        ref_log = run_child(1 if SELF_SPEC else 0, ref_path, num_spec=0 if SELF_SPEC else None, tag=ref_label)
        with open(ref_path, encoding="utf-8") as f:
            base = json.load(f)
        with open(test_path, encoding="utf-8") as f:
            qfa = json.load(f)

    print("== run markers ==")
    ref_marks = report_log(ref_label, ref_log)
    test_marks = report_log(test_label, test_log)
    if not test_marks["paged_engaged"]:
        print("[RED] the QFA run never entered the paged path - the comparison is vacuous")
        return 1
    if SELF_SPEC and not ref_marks["paged_engaged"]:
        print("[RED] the spec0 reference never entered the paged path - it is not the same path")
        return 1

    all_ok = True
    for i, (b, q) in enumerate(zip(base, qfa)):
        b_ids, q_ids = b["token_ids"], q["token_ids"]
        prefix = 0
        for x, y in zip(b_ids, q_ids):
            if x != y:
                break
            prefix += 1
        shared = min(len(b_ids), len(q_ids))
        width = max(len(ref_label), len(test_label))
        print(f"== prompt {i}: {PROMPTS[i]!r}")
        print(f"  {(ref_label + ':').ljust(width + 1)} {b['text']!r}")
        print(f"  {(test_label + ':').ljust(width + 1)} {q['text']!r}")
        note = "informational - a near-tie flips this at any step"
        print(f"  greedy_prefix_match={prefix}/{shared} ({note})")
        stats = report_numerics(b, q, prefix, ref_label, test_label)
        overlap_gate = OVERLAP_GATE * TOP_LOGPROBS
        ok = (
            stats["steps"] > 0
            and stats["max_delta"] <= DELTA_GATE
            and stats["mean_overlap"] >= overlap_gate
        )
        print(
            f"  [{'GREEN' if ok else 'RED'}] prompt {i}: "
            f"delta {stats['max_delta']:.4f} vs gate {DELTA_GATE}, "
            f"overlap {stats['mean_overlap']:.2f} vs gate {overlap_gate:.1f}"
        )
        all_ok = all_ok and ok

    print(f"[{'GREEN' if all_ok else 'RED'}] QFA MXFP8 decode vs BF16 baseline overall")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
