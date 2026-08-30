#!/usr/bin/env python3
"""Measure what QFA costs and saves against the FIA baseline.

    python3 scripts/bench_qfa_vs_fia.py --auto

That brings up 27B.sh twice -- baseline with QFA=0 NO_PREFIX_CACHE=1, candidate
with QFA=1 -- measures each, takes it down, and prints the table. Keeping the
two configurations symmetric is the script's job, not the operator's: 27B.sh
ties --no-enable-prefix-caching to QFA=1, so a baseline started by hand differs
by more than the attention operator unless someone remembers that flag.
Anything else you pass through (GPU_MEM_UTIL, GRAPH, MTP, MODEL_PATH...) is
inherited by both runs from this process's environment.

The pieces are still usable on their own against servers you manage:

    python3 scripts/bench_qfa_vs_fia.py --label fia --server-log fia.log --out fia.json
    python3 scripts/bench_qfa_vs_fia.py --compare fia.json qfa.json

--compare fingerprints both server logs and says so when the two runs disagree
on anything but QFA, so that path cannot quietly produce a meaningless table
either. Every prompt also carries a unique prefix, so prefix caching cannot help
either run whatever the flags say.

Nothing is written outside the working directory (the server's root partition is
tight). The script is non-interactive and exits non-zero if a server never
answers or the two runs are not comparable.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# One filler sentence repeated to reach the requested prompt length. Chinese so
# the token count per repetition stays stable across the tokenizer's paths.
FILLER = "请仔细阅读下面这段材料，并在最后给出一段完整的总结与分析。"
QUESTION = "\n\n请用中文写一段约两百字的总结。"


def _post_stream(url: str, payload: dict, timeout: float) -> tuple[float, float, int]:
    """Send one streaming completion. Returns (ttft, total, output_tokens)."""
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    ttft = float("nan")
    output_tokens = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[len("data:") :].strip()
            if chunk == "[DONE]":
                break
            event = json.loads(chunk)
            usage = event.get("usage")
            if usage:
                output_tokens = usage.get("completion_tokens", output_tokens)
            for choice in event.get("choices", []):
                if choice.get("delta", {}).get("content"):
                    if ttft != ttft:  # still NaN: this is the first token
                        ttft = time.perf_counter() - start
    return ttft, time.perf_counter() - start, output_tokens


def _build_prompt(index: int, repeats: int) -> str:
    # The unique head is what keeps prefix caching out of the measurement.
    return f"[样本 {index:04d}] " + FILLER * repeats + QUESTION


def _scrape_metrics(base: str) -> dict[str, float]:
    """Pull the gauges we care about out of the Prometheus endpoint."""
    wanted = ("vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting")
    out: dict[str, float] = {}
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=5) as response:
            text = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return out
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for name in wanted:
            if line.startswith(name):
                try:
                    value = float(line.rsplit(" ", 1)[1])
                except (IndexError, ValueError):
                    continue
                out[name] = max(out.get(name, value), value)
    return out


def _parse_server_log(path: str) -> dict[str, float]:
    """Pick the KV cache facts out of a captured server log.

    These are the numbers that answer "did the cache get smaller"; they are
    printed once at startup and do not need any load to observe.
    """
    patterns = {
        "kv_cache_tokens": r"GPU KV cache size:\s*([\d,]+)\s*tokens",
        "max_concurrency": r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x",
        "kv_cache_gib": r"Available KV cache memory:\s*([\d.]+)\s*GiB",
    }
    facts: dict[str, float] = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError as exc:
        print(f"WARNING: cannot read server log {path}: {exc}", file=sys.stderr)
        return facts
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            facts[key] = float(match.group(1).replace(",", ""))
        else:
            print(f"WARNING: {key} not found in {path}", file=sys.stderr)
    return facts


# Scalars vLLM prints in its startup config line. Two runs that differ on any of
# these are not a QFA-vs-FIA comparison, whatever the table says.
FINGERPRINT = {
    "enable_prefix_caching": r"enable_prefix_caching=(\w+)",
    "enforce_eager": r"enforce_eager=(\w+)",
    "kv_cache_dtype": r"kv_cache_dtype=([\w.']+)",
    "tensor_parallel_size": r"tensor_parallel_size=(\d+)",
    "max_seq_len": r"max_seq_len=(\d+)",
    "cudagraph_mode": r"['\"]cudagraph_mode['\"]\s*:\s*['\"](\w+)['\"]",
}


def _fingerprint(path: str) -> dict[str, str]:
    """What the server was actually started with, read back from its log."""
    out: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return out
    for key, pattern in FINGERPRINT.items():
        match = re.search(pattern, text)
        if match:
            out[key] = match.group(1)
    # 27B.sh announces this on stderr, which lands in the same log.
    out["qfa"] = "1" if "QFA on:" in text else "0"
    return out


def _wait_healthy(base: str, proc: subprocess.Popen, log_path: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode} during startup; see {log_path}")
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(3)
    raise RuntimeError(f"server was not healthy within {timeout}s; see {log_path}")


def _stop(proc: subprocess.Popen) -> None:
    """Take down the whole process group: 27B.sh execs vllm, which forks workers."""
    if proc.poll() is not None:
        return
    group = os.getpgid(proc.pid)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=60)
            return
        except subprocess.TimeoutExpired:
            continue


def _wait_port_free(base: str, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=3):
                pass
        except (urllib.error.URLError, TimeoutError, OSError):
            return
        time.sleep(2)
    print("WARNING: previous server still answering; the next run may attach to it", file=sys.stderr)


@contextlib.contextmanager
def serve(args: argparse.Namespace, env_extra: dict[str, str], log_path: str):
    base = f"http://127.0.0.1:{args.port}"
    env = dict(os.environ, **env_extra)
    env["VLLM_PORT"] = str(args.port)
    shown = " ".join(f"{k}={v}" for k, v in sorted(env_extra.items()))
    print(f"[auto] starting {args.serve_script} with {shown} -> {log_path}", file=sys.stderr)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            ["bash", args.serve_script],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_healthy(base, proc, log_path, args.startup_timeout)
            print("[auto] server is up", file=sys.stderr)
            yield
        finally:
            print("[auto] stopping server", file=sys.stderr)
            _stop(proc)
    _wait_port_free(base)


def run_auto(args: argparse.Namespace) -> int:
    """Both runs, back to back, with the configurations held symmetric."""
    plan = [("fia", {"QFA": "0", "NO_PREFIX_CACHE": "1"}), ("qfa", {"QFA": "1"})]
    paths = []
    for label, env_extra in plan:
        log_path, out_path = f"{label}.log", f"{label}.json"
        args.label, args.server_log = label, log_path
        with serve(args, env_extra, log_path):
            report = measure(args)
        with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"[auto] wrote {out_path}", file=sys.stderr)
        paths.append(out_path)
    return compare(paths)


def measure(args: argparse.Namespace) -> dict:
    base = f"http://127.0.0.1:{args.port}"
    url = f"{base}/v1/chat/completions"

    def one(index: int) -> tuple[float, float, int]:
        payload = {
            "model": args.model,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": _build_prompt(index, args.prompt_repeats)}],
        }
        return _post_stream(url, payload, args.timeout)

    print(f"[{args.label}] warmup: {args.warmup} request(s)", file=sys.stderr)
    try:
        for i in range(args.warmup):
            one(-1 - i)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"ERROR: server at {url} did not answer: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    peak: dict[str, float] = {}
    stop = threading.Event()

    def sampler() -> None:
        while not stop.wait(args.sample_interval):
            for key, value in _scrape_metrics(base).items():
                peak[key] = max(peak.get(key, value), value)

    watcher = threading.Thread(target=sampler, daemon=True)
    watcher.start()

    print(
        f"[{args.label}] {args.requests} request(s), concurrency {args.concurrency}",
        file=sys.stderr,
    )
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one, range(args.requests)))
    wall = time.perf_counter() - wall_start
    stop.set()
    watcher.join(timeout=2 * args.sample_interval + 5)

    ttfts = [t for t, _, _ in results if t == t]
    totals = [d for _, d, _ in results]
    tokens = [n for _, _, n in results]
    decode_rates = [
        (n - 1) / (d - t) for t, d, n in results if t == t and n > 1 and d > t
    ]
    report = {
        "label": args.label,
        "workload": {
            "requests": args.requests,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "prompt_repeats": args.prompt_repeats,
        },
        "server": _parse_server_log(args.server_log) if args.server_log else {},
        "config": _fingerprint(args.server_log) if args.server_log else {},
        "peak_metrics": peak,
        "results": {
            "wall_seconds": wall,
            "output_tokens_total": sum(tokens),
            "output_tokens_per_s": sum(tokens) / wall if wall else 0.0,
            "ttft_mean_s": statistics.fmean(ttfts) if ttfts else None,
            "ttft_p90_s": (sorted(ttfts)[int(0.9 * (len(ttfts) - 1))] if ttfts else None),
            "total_mean_s": statistics.fmean(totals) if totals else None,
            "decode_tokens_per_s_mean": (statistics.fmean(decode_rates) if decode_rates else None),
        },
    }
    if len(ttfts) < args.requests:
        report["results"]["requests_without_tokens"] = args.requests - len(ttfts)
    return report


ROWS = [
    ("KV cache capacity (tokens)", ("server", "kv_cache_tokens"), "higher"),
    ("KV cache memory (GiB)", ("server", "kv_cache_gib"), "note"),
    ("Max concurrency (x)", ("server", "max_concurrency"), "higher"),
    ("Peak KV usage (%)", ("peak_metrics", "vllm:kv_cache_usage_perc"), "note"),
    ("Throughput (out tok/s)", ("results", "output_tokens_per_s"), "higher"),
    ("Decode rate (tok/s/req)", ("results", "decode_tokens_per_s_mean"), "higher"),
    ("TTFT mean (s)", ("results", "ttft_mean_s"), "lower"),
    ("TTFT p90 (s)", ("results", "ttft_p90_s"), "lower"),
    ("Wall clock (s)", ("results", "wall_seconds"), "lower"),
]


def _check_comparable(baseline: dict, candidate: dict, name_a: str, name_b: str) -> bool:
    """Say plainly when the two runs differ in more -- or less -- than QFA.

    27B.sh ties --no-enable-prefix-caching to QFA=1, so a hand-started baseline
    is asymmetric unless it was given NO_PREFIX_CACHE=1. --auto handles that;
    this catches the case where nobody did.
    """
    left = baseline.get("config", {})
    right = candidate.get("config", {})
    if not left or not right:
        print(
            "NOTE: no server log for at least one run, so the two configurations could not be"
            " checked. Pass --server-log to both, or use --auto.",
            file=sys.stderr,
        )
        return True

    ok = True
    if left.get("qfa") == right.get("qfa"):
        print(
            f"ERROR: both runs have QFA={left.get('qfa')}; there is nothing to compare.",
            file=sys.stderr,
        )
        ok = False
    for key in FINGERPRINT:
        a, b = left.get(key), right.get(key)
        if a is not None and b is not None and a != b:
            print(f"ERROR: {key} differs -- {name_a}={a} {name_b}={b}", file=sys.stderr)
            ok = False
    if not ok:
        print(
            "The table below compares runs that differ in more than the attention operator."
            " Re-run with --auto, which keeps them symmetric.",
            file=sys.stderr,
        )
    return ok


def compare(paths: list[str]) -> int:
    reports = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            reports.append(json.load(handle))
    baseline, candidate = reports
    name_a, name_b = baseline.get("label", paths[0]), candidate.get("label", paths[1])

    comparable = _check_comparable(baseline, candidate, name_a, name_b)

    print(f"\n{'metric':<28}{name_a:>16}{name_b:>16}{'delta':>14}")
    print("-" * 74)
    deltas: dict[str, float] = {}
    for title, (section, key), _better in ROWS:
        a = baseline.get(section, {}).get(key)
        b = candidate.get(section, {}).get(key)
        if a is None or b is None:
            print(f"{title:<28}{'-' if a is None else f'{a:.2f}':>16}{'-' if b is None else f'{b:.2f}':>16}{'':>14}")
            continue
        delta = (b - a) / a * 100 if a else float("nan")
        deltas[title] = delta
        print(f"{title:<28}{a:>16.2f}{b:>16.2f}{delta:>13.1f}%")

    print("\nverdict")
    capacity = deltas.get("KV cache capacity (tokens)")
    if capacity is None:
        print("  KV capacity : SKIPPED  (pass --server-log on both runs to get it)")
    elif capacity > 5:
        print(f"  KV capacity : GREEN    {capacity:+.1f}% -- the cache got smaller, MXFP8 storage is live")
    elif capacity > -5:
        print(f"  KV capacity : EXPECTED {capacity:+.1f}% -- parity, the cache is still bf16")
    else:
        print(f"  KV capacity : RED      {capacity:+.1f}% -- QFA should never cost capacity")

    throughput = deltas.get("Throughput (out tok/s)")
    if throughput is None:
        print("  Throughput  : SKIPPED")
    elif throughput > -5:
        print(f"  Throughput  : GREEN    {throughput:+.1f}%")
    else:
        print(
            f"  Throughput  : EXPECTED {throughput:+.1f}% -- QFA quantizes the whole KV cache every"
            " step until the cache is stored as MXFP8"
        )
    print()
    return 0 if comparable else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--auto",
        action="store_true",
        help="run both configurations end to end (starts and stops the server itself)",
    )
    parser.add_argument("--serve-script", default="scripts/27B.sh", help="server launcher for --auto")
    parser.add_argument("--startup-timeout", type=float, default=1800.0, help="--auto: wait for /health")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"), help="print a table from two result files")
    parser.add_argument("--label", default="run", help="name for this run, shown in the comparison")
    parser.add_argument("--out", help="write the result JSON here")
    parser.add_argument("--server-log", help="captured server stdout, for the KV cache numbers")
    parser.add_argument("--port", type=int, default=6969)
    parser.add_argument("--model", default="qwen3.8")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt-repeats", type=int, default=40, help="filler sentences per prompt")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    args = parser.parse_args()

    if args.compare:
        return compare(args.compare)
    if args.auto:
        return run_auto(args)

    report = measure(args)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")
        print(f"[{args.label}] wrote {args.out}", file=sys.stderr)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
