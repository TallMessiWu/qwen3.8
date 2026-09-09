#!/usr/bin/env python3
"""Measure MTP speculative-decoding acceptance against a running vLLM server.

Why this exists as a script rather than a curl one-liner: ``/metrics`` only
carries monotonic counters, so a bare scrape mixes in every request the server
has ever served -- including the warm-up and whatever the previous experiment
left behind. Reading the two-decimal ratio out of the log line is worse still,
because "0.00" and "0.004" print the same and the third draft position is
exactly where the interesting difference sits. So: snapshot the counters, drive
a fixed prompt set, snapshot again, and report the delta as absolute counts.

The verdict is the process exit code (0 GREEN / 1 RED / 2 harness error), so a
caller can branch on it without parsing stdout. Nothing is piped internally.

Determinism: temperature 0, fixed prompts, fixed max_tokens, one request at a
time by default. Acceptance depends on the batch shape, so --concurrency must
be held equal across the two sides of any A/B.

Usage
-----
    python3 scripts/checks/mtp_accept_rate.py --port 6969 --label full-graph
    python3 scripts/checks/mtp_accept_rate.py --port 6969 --label draft-eager \
        --json draft-eager.json --min-accept-per-draft 1.5

Compare two runs:
    python3 scripts/checks/mtp_accept_rate.py --compare a.json b.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# Counter base names; Prometheus appends ``_total`` to every counter.
_C_DRAFTS = "vllm:spec_decode_num_drafts_total"
_C_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens_total"
_C_ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
_C_PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"

# A fixed, self-contained prompt set. Deliberately mixed: a short factual ask,
# a list, a bit of code and a short chain of reasoning. Acceptance rate is
# strongly content-dependent -- highly predictable text drafts well, novel text
# does not -- so the set has to stay frozen for numbers to be comparable across
# runs. Do not "improve" these prompts; add new ones behind a new --prompt-set.
_PROMPTS_V1 = [
    "用一句话解释什么是推测解码（speculative decoding）。",
    "列出快速排序的三个步骤，每步一行。",
    "写一个 Python 函数 fib(n)，返回第 n 个斐波那契数，用迭代实现。",
    "如果一列火车以每小时 60 公里的速度行驶 2.5 小时，它走了多远？给出计算过程。",
    "Explain in two sentences why matrix multiplication is not commutative.",
    "把下面这句话翻译成英文：昇腾 NPU 上的算子需要先经过图捕获才能重放。",
]

_METRIC_LINE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[^\s]+)$")
_LABEL_KV = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


class HarnessError(RuntimeError):
    """Something went wrong with the harness itself, not with the model."""


@dataclass
class SpecCounters:
    """One ``/metrics`` snapshot, summed over engines."""

    drafts: float = 0.0
    draft_tokens: float = 0.0
    accepted: float = 0.0
    per_pos: dict[int, float] = field(default_factory=dict)

    def minus(self, other: SpecCounters) -> SpecCounters:
        positions = set(self.per_pos) | set(other.per_pos)
        return SpecCounters(
            drafts=self.drafts - other.drafts,
            draft_tokens=self.draft_tokens - other.draft_tokens,
            accepted=self.accepted - other.accepted,
            per_pos={p: self.per_pos.get(p, 0.0) - other.per_pos.get(p, 0.0) for p in sorted(positions)},
        )


def _http(url: str, payload: dict | None, timeout: float) -> str:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def scrape(base: str, timeout: float) -> SpecCounters:
    """Sum the spec-decode counters across every engine label set."""
    try:
        text = _http(f"{base}/metrics", None, timeout)
    except (urllib.error.URLError, OSError) as exc:
        raise HarnessError(f"cannot reach {base}/metrics: {exc}") from exc

    out = SpecCounters()
    seen = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE.match(line)
        if not m:
            continue
        name = m.group("name")
        if not name.startswith("vllm:spec_decode_"):
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        labels = dict(_LABEL_KV.findall(m.group("labels") or ""))
        seen = True
        if name == _C_DRAFTS:
            out.drafts += value
        elif name == _C_DRAFT_TOKENS:
            out.draft_tokens += value
        elif name == _C_ACCEPTED:
            out.accepted += value
        elif name == _C_PER_POS:
            pos = labels.get("position")
            if pos is not None:
                out.per_pos[int(pos)] = out.per_pos.get(int(pos), 0.0) + value

    if not seen:
        raise HarnessError(
            "no vllm:spec_decode_* metrics exposed. The server is either running "
            "without --speculative-config (MTP=0) or was started with "
            "--disable-log-stats."
        )
    return out


def one_request(base: str, model: str, prompt: str, endpoint: str, max_tokens: int, timeout: float) -> dict:
    if endpoint == "chat":
        url = f"{base}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            # Pinned here rather than inherited from the server's THINKING env
            # so the harness measures the same thing on every launcher config.
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "stream": False,
            "seed": 0,
        }
    else:
        url = f"{base}/v1/completions"
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "stream": False,
            "seed": 0,
        }
    body = _http(url, payload, timeout)
    return json.loads(body)


def _completion_tokens(resp: dict) -> int:
    return int(resp.get("usage", {}).get("completion_tokens", 0))


def _text_of(resp: dict, endpoint: str) -> str:
    choices = resp.get("choices") or [{}]
    if endpoint == "chat":
        return (choices[0].get("message") or {}).get("content") or ""
    return choices[0].get("text") or ""


def _measure(args, base: str, prompts: list[str], max_tokens: int) -> tuple[SpecCounters, int, list[str], float]:
    """Run the prompt set once at ``max_tokens`` and return the counter delta."""
    before = scrape(base, args.timeout)
    t0 = time.monotonic()

    texts: list[str] = []
    total_completion = 0

    def run(prompt: str) -> dict:
        return one_request(base, args.model, prompt, args.endpoint, max_tokens, args.timeout)

    for _ in range(args.repeats):
        if args.concurrency <= 1:
            responses = [run(p) for p in prompts]
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                responses = list(pool.map(run, prompts))
        for resp in responses:
            total_completion += _completion_tokens(resp)
            texts.append(_text_of(resp, args.endpoint))

    elapsed = time.monotonic() - t0
    after = scrape(base, args.timeout)
    return after.minus(before), total_completion, texts, elapsed


def _warmup(args, base: str, prompts: list[str]) -> None:
    # Outside every measured window: a fresh server's first decode can still be
    # paying for lazy compilation, and those drafts would land in the delta.
    for prompt in prompts[: args.warmup]:
        one_request(base, args.model, prompt, args.endpoint, min(32, args.max_tokens), args.timeout)


def drive(args) -> tuple[SpecCounters, int, list[str]]:
    base = f"http://{args.host}:{args.port}"
    prompts = list(_PROMPTS_V1)
    _warmup(args, base, prompts)
    delta, total_completion, texts, elapsed = _measure(args, base, prompts, args.max_tokens)

    if delta.drafts <= 0:
        raise HarnessError(
            "the spec-decode counters did not move. Either every request hit "
            "prefix cache / returned instantly, or speculative decoding is off."
        )
    if total_completion <= 0:
        raise HarnessError("the server returned zero completion tokens; nothing was measured.")

    print(f"wall clock {elapsed:.1f}s, {total_completion} completion tokens")
    return delta, total_completion, texts


def sweep(args, stops: list[int]) -> int:
    """Acceptance resolved by position in the generated sequence.

    Aggregate acceptance cannot tell "the draft is uniformly a bit wrong" apart
    from "the first few steps after a prefill are very wrong": a short run is
    mostly early steps, a long run mostly late ones, and the two hypotheses
    predict the same average. So run the same prompt set at an increasing
    max_tokens and subtract consecutive runs. Greedy decoding makes every run
    regenerate the identical prefix, so the difference between the K1 and K2
    runs is exactly the work spent on output positions [K1, K2).

    Each band therefore reports the acceptance the draft actually achieved in
    that slice of the sequence, using nothing but the existing counters.
    """
    base = f"http://{args.host}:{args.port}"
    prompts = list(_PROMPTS_V1)
    _warmup(args, base, prompts)

    rows = []
    prev_stop = 0
    prev = SpecCounters()
    for stop in stops:
        delta, completion, _, elapsed = _measure(args, base, prompts, stop)
        band = delta.minus(prev)
        rows.append((prev_stop, stop, band, delta, completion, elapsed))
        prev_stop, prev = stop, delta

    print()
    print(f"=== acceptance by output position [{args.label}] ===")
    print(f"{'band':<14}{'drafts':>10}{'accepted':>10}{'acc/draft':>12}{'pos0':>9}{'pos1':>9}{'pos2':>9}")
    ok = False
    for lo, hi, band, _, _, _ in rows:
        if band.drafts <= 0:
            print(f"{f'[{lo},{hi})':<14}{0:>10}{'':>10}{'  (no new drafts: generation already finished)':<}")
            continue
        ok = True
        per_draft = band.accepted / band.drafts
        cells = []
        prevpos = band.drafts
        for pos in sorted(band.per_pos):
            count = band.per_pos[pos]
            cells.append(f"{(count / prevpos if prevpos else 0.0):>9.3f}")
            prevpos = count
        while len(cells) < 3:
            cells.append(f"{'':>9}")
        print(f"{f'[{lo},{hi})':<14}{band.drafts:>10.0f}{band.accepted:>10.0f}{per_draft:>12.4f}" + "".join(cells[:3]))

    if not ok:
        print("\nno band produced drafts; lower the first stop or raise --repeats")
        return 2
    print("\npos columns are conditional acceptance within the band.")
    print("Flat columns mean a uniform per-step error; a rising curve means the")
    print("damage is concentrated in the steps right after a prefill.")
    return 0


def report(delta: SpecCounters, total_completion: int, label: str, num_spec: int | None) -> dict:
    drafts = delta.drafts
    positions = sorted(delta.per_pos)
    accept_per_draft = delta.accepted / drafts if drafts else 0.0
    accept_rate = delta.accepted / delta.draft_tokens if delta.draft_tokens else 0.0

    print()
    print(f"=== MTP acceptance [{label}] ===")
    print(f"{'drafts':<28}{drafts:>12.0f}")
    print(f"{'draft tokens':<28}{delta.draft_tokens:>12.0f}")
    print(f"{'accepted tokens':<28}{delta.accepted:>12.0f}")
    print(f"{'accepted / draft':<28}{accept_per_draft:>12.4f}")
    print(f"{'accepted / draft token':<28}{accept_rate:>12.4f}")
    print()
    # per_pos[k] counts drafts where positions 0..k were ALL accepted (see
    # SpecDecodingStats.observe_draft: it increments 0..num_accepted-1), so the
    # per-position column is a survival curve, and the conditional column is
    # what actually isolates which draft step falls over.
    print(f"{'pos':<6}{'accepted (abs)':>16}{'cumulative':>14}{'conditional':>14}")
    prev = drafts
    per_pos_out = []
    for pos in positions:
        count = delta.per_pos[pos]
        cumulative = count / drafts if drafts else 0.0
        conditional = count / prev if prev else 0.0
        print(f"{pos:<6}{count:>16.0f}{cumulative:>14.4f}{conditional:>14.4f}")
        per_pos_out.append({"position": pos, "accepted": count, "cumulative": cumulative, "conditional": conditional})
        prev = count

    if num_spec is not None and len(positions) != num_spec:
        print(f"WARNING: server reports {len(positions)} draft positions, expected {num_spec}")

    return {
        "label": label,
        "drafts": drafts,
        "draft_tokens": delta.draft_tokens,
        "accepted": delta.accepted,
        "accepted_per_draft": accept_per_draft,
        "accepted_per_draft_token": accept_rate,
        "completion_tokens": total_completion,
        "per_pos": per_pos_out,
    }


def compare(path_a: str, path_b: str) -> int:
    with open(path_a) as fa, open(path_b) as fb:
        a, b = json.load(fa), json.load(fb)
    print(f"=== {a['label']}  vs  {b['label']} ===")
    print(f"{'metric':<26}{a['label']:>16}{b['label']:>16}{'delta':>12}")
    for key in ("drafts", "accepted", "accepted_per_draft", "accepted_per_draft_token"):
        va, vb = a[key], b[key]
        print(f"{key:<26}{va:>16.4f}{vb:>16.4f}{vb - va:>12.4f}")
    print()
    print(f"{'pos':<6}{'conditional A':>16}{'conditional B':>16}{'delta':>12}")
    by_pos_a = {p["position"]: p for p in a["per_pos"]}
    for entry in b["per_pos"]:
        pos = entry["position"]
        ca = by_pos_a.get(pos, {}).get("conditional", 0.0)
        print(f"{pos:<6}{ca:>16.4f}{entry['conditional']:>16.4f}{entry['conditional'] - ca:>12.4f}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6969)
    p.add_argument("--model", default="qwen3.8", help="--served-model-name of the running server")
    p.add_argument("--endpoint", choices=("chat", "completions"), default="chat")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--repeats", type=int, default=1, help="how many times to run the whole prompt set")
    p.add_argument("--warmup", type=int, default=2, help="prompts to run before the counters are snapshotted")
    p.add_argument("--concurrency", type=int, default=1, help="hold this equal across both sides of an A/B")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--label", default="run")
    p.add_argument("--json", dest="json_out", default=None, help="write the result here for a later --compare")
    p.add_argument("--num-spec", type=int, default=None, help="expected num_speculative_tokens, for a sanity warning")
    p.add_argument(
        "--min-accept-per-draft",
        type=float,
        default=None,
        help="GREEN when accepted/draft is at least this. Omit to only report.",
    )
    p.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), default=None)
    p.add_argument(
        "--sweep",
        default=None,
        metavar="K1,K2,...",
        help="ascending max_tokens stops; reports acceptance per output-position band",
    )
    args = p.parse_args()

    if args.compare:
        return compare(*args.compare)

    if args.sweep:
        try:
            stops = [int(s) for s in args.sweep.split(",") if s.strip()]
        except ValueError:
            print("--sweep takes a comma-separated list of integers", file=sys.stderr)
            return 2
        if len(stops) < 2 or stops != sorted(stops) or len(set(stops)) != len(stops):
            print("--sweep needs at least two strictly increasing stops", file=sys.stderr)
            return 2
        try:
            return sweep(args, stops)
        except HarnessError as exc:
            print(f"HARNESS ERROR: {exc}", file=sys.stderr)
            return 2
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            print(f"HARNESS ERROR: request failed: {exc}", file=sys.stderr)
            return 2

    try:
        delta, total_completion, texts = drive(args)
    except HarnessError as exc:
        print(f"HARNESS ERROR: {exc}", file=sys.stderr)
        return 2
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        print(f"HARNESS ERROR: request failed: {exc}", file=sys.stderr)
        return 2

    result = report(delta, total_completion, args.label, args.num_spec)

    # A broken draft path and a broken target model both drag the acceptance
    # rate down, and only one of them also wrecks the text. Print a sample so
    # the reader can tell those two apart without a second run.
    print()
    print("--- first response (truncated) ---")
    print((texts[0] if texts else "")[:400])

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.json_out}")

    if args.min_accept_per_draft is None:
        print("\nno threshold given (--min-accept-per-draft); reporting only")
        return 0

    if result["accepted_per_draft"] >= args.min_accept_per_draft:
        print(f"\nGREEN: accepted/draft {result['accepted_per_draft']:.4f} >= {args.min_accept_per_draft}")
        return 0
    print(f"\nRED: accepted/draft {result['accepted_per_draft']:.4f} < {args.min_accept_per_draft}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
