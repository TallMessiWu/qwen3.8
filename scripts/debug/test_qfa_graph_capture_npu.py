#!/usr/bin/env python3
"""Burn down what the QFA graph path (6.4) added, without starting the engine.

test_junlin_qfa_npu.py's GRAPH case already proves the D4 shape for the *main
op*: metadata computed outside into a fixed buffer, quantized payloads prepared
on the host, capture, replay, in-place batch swap. The engine does something it
does not cover -- it captures the **quantization** too, because under
FULL_DECODE_ONLY q is produced by the in-graph QKV projection and the current
step's K/V is written into the cache in-graph by reshape_and_cache. Neither can
be hoisted out before the cache is stored as MXFP8.

So these cases exercise the real functions from the plugin
(_qfa_quant / _qfa_quant_v) inside a capture, in the exact arrangement
full_graph_qfa + _update_qfa_graph_buffers use:

  VDESCALE-STUB  paged metadata built with a placeholder v_descale vs the real
                 one -> identical plan and identical output. quantMode=1 refuses
                 a null v_descale at the aclnn entry, but under PA_BBND nothing
                 reads it (the rank check covers only TND/PA_BNBD/PA_NZ, and the
                 AICPU's dim0 check is guarded on TND). _attach_qfa_inputs leans
                 on that to build the plan before any layer has quantized
                 anything, which is what makes the graph path possible at all.
  QUANT-IN-GRAPH capture npu_dynamic_mx_quant + the main op, with the length /
                 plan / block-table buffers allocated *inside* the capture and
                 only ever filled from outside, gated by an ExternalEvent:
                 A replay with unchanged inputs == eager,
                 B rewrite the bf16 cache, q and buffers in place, refresh
                   outside, replay == eager on the new batch.
                 B is the one that matters: it proves the captured quantization
                 re-reads the cache instead of baking capture-time values.
  MAXKV          same, with the capture constant max_seqlen_kv raised to
                 --max-model-len (133120 by default). It is baked into the
                 captured op and feeds AdjustSinnerAndSouter, so metadata and
                 the main op must agree on it -- this asks whether the split
                 plan still fits the op's fixed 4096-int32 output at that value.
  POOL-MEM       reports the allocator delta across a capture at --blocks worth
                 of cache, so "does the in-graph whole-cache quantization fit"
                 gets a number instead of a guess. Reports only; never RED.

Each case runs in its own subprocess: a device abort poisons everything after
it, so a shared process could only ever report the first failure. A hang (a
mis-ordered event) is caught by --case-timeout rather than wedging the run.

Usage (inside the serving container, after pip_install_qfa.sh):
  python scripts/debug/test_qfa_graph_capture_npu.py
  python scripts/debug/test_qfa_graph_capture_npu.py --cases QUANT-IN-GRAPH
  python scripts/debug/test_qfa_graph_capture_npu.py --cases POOL-MEM --blocks 9672
"""

from __future__ import annotations

import argparse
import ast
import builtins
import importlib.util
import math
import os
import pathlib
import subprocess
import sys

import torch

CASES = ("VDESCALE-STUB", "QUANT-IN-GRAPH", "MAXKV", "POOL-MEM")


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------
QUANT_HELPERS = ("_qfa_quant", "_qfa_quant_v")


def find_attention_v1(override: str | None) -> pathlib.Path:
    """Locate attention_v1.py without executing the package."""
    if override:
        return pathlib.Path(override)
    spec = importlib.util.find_spec("vllm_ascend")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("vllm_ascend is not importable; pass --attention-v1")
    root = pathlib.Path(list(spec.submodule_search_locations)[0])
    return root / "attention" / "attention_v1.py"


def load_quant_helpers(path: pathlib.Path):
    """Lift _qfa_quant/_qfa_quant_v out of attention_v1.py's source.

    Importing the module instead pulls in the plugin's whole graph
    (device_op -> ops/__init__ -> moe_mlp -> device_op) and trips a circular
    import outside the order the engine happens to import things in. Both
    helpers are plain torch/torch_npu code, so the file's own source is lifted
    straight out: the test cannot drift from what the engine runs, and nothing
    else gets imported. The guard below is what makes that safe -- if either
    helper ever grows a dependency on the rest of the plugin, this says so
    instead of quietly testing something stale.
    """
    import torch_npu

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in QUANT_HELPERS]
    missing = set(QUANT_HELPERS) - {n.name for n in nodes}
    if missing:
        raise RuntimeError(f"{path} no longer defines {sorted(missing)} at module level")

    allowed = {"torch", "torch_npu"} | set(dir(builtins))
    for node in nodes:
        bound = {arg.arg for arg in node.args.args}
        bound |= {i.id for i in ast.walk(node) if isinstance(i, ast.Name) and isinstance(i.ctx, ast.Store)}
        for inner in ast.walk(node):
            if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Load):
                if inner.id not in bound and inner.id not in allowed:
                    raise RuntimeError(
                        f"{node.name} now references {inner.id!r} from attention_v1; lifting it out "
                        "of the file no longer works -- import the module instead and fix the cycle"
                    )

    namespace: dict = {"torch": torch, "torch_npu": torch_npu}
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    print(f"[OK] lifted {list(QUANT_HELPERS)} from {path}")
    return namespace[QUANT_HELPERS[0]], namespace[QUANT_HELPERS[1]]


def bootstrap(args):
    """Register the custom ops, then hand back the plugin's own quant helpers."""
    import torch_npu  # noqa: F401

    try:
        from vllm_ascend.utils import bootstrap_custom_op_env

        bootstrap_custom_op_env(include_vendor_lib=True)
    except Exception:
        import vllm_ascend

        vendor = os.path.join(
            os.path.dirname(vllm_ascend.__file__), "_cann_ops_custom", "vendors", "custom_transformer"
        )
        if os.path.isdir(vendor):
            prev = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = vendor + (":" + prev if prev else "")
        else:
            print(f"[WARN] custom opp vendor dir missing: {vendor}")
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    for name in ("npu_quant_flash_attn", "npu_quant_flash_attn_metadata"):
        assert hasattr(torch.ops._C_ascend, name), f"{name} not registered"

    return load_quant_helpers(find_attention_v1(args.attention_v1))


def v_descale_stub() -> torch.Tensor:
    """What _attach_qfa_inputs hands the metadata op: non-null, PA_BBND's rank,
    minimal extents. e8m0 is built through a uint8 view -- torch cannot fill a
    float8 tensor directly."""
    return torch.zeros(1, 1, 1, 1, 2, dtype=torch.uint8).npu().view(torch.float8_e8m0fnu)


def causal_mask() -> torch.Tensor:
    # triu(diagonal=1) int8, 1 = masked future: what attention_v1 hands FIA and
    # what QFA's mask_mode=3 expects. The doc example's tril is wrong.
    return torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1).npu()


# --------------------------------------------------------------------------
# one decode step, in the shapes the engine actually passes
# --------------------------------------------------------------------------
class Step:
    """A decode batch in engine layout: bf16 paged cache + bf16 q."""

    def __init__(self, args, seed: int):
        torch.manual_seed(seed)
        self.nq, self.nkv, self.d = args.num_heads, args.num_kv_heads, args.head_dim
        self.block_size = args.block_size
        self.kv_lens = [int(x) for x in args.kv_lens.split(",")]
        self.b = len(self.kv_lens)
        self.q_len = args.q_len
        self.scale = 1.0 / math.sqrt(self.d)

        cols = max(1, (max(self.kv_lens) + self.block_size - 1) // self.block_size)
        needed = self.b * cols
        self.num_blocks = max(args.blocks, needed)

        dev = "npu"
        # _get_fia_params hands attention the cache as (num_blocks, block_size,
        # N * D); _qfa_paged_call is what splits the heads back out.
        self.k_cache = torch.randn(
            self.num_blocks, self.block_size, self.nkv * self.d, dtype=torch.bfloat16, device=dev
        )
        self.v_cache = torch.randn_like(self.k_cache)
        self.q = torch.randn(self.b * self.q_len, self.nq, self.d, dtype=torch.bfloat16, device=dev)

        table = torch.zeros(self.b, cols, dtype=torch.int32)
        for i in range(self.b):
            table[i] = torch.arange(i * cols, (i + 1) * cols, dtype=torch.int32)
        self.block_table = table.to(dev)
        self.cu_seqlens_q = torch.tensor(
            [0] + [(i + 1) * self.q_len for i in range(self.b)], dtype=torch.int32, device=dev
        )
        self.seqused_kv = torch.tensor(self.kv_lens, dtype=torch.int32, device=dev)
        self.max_seqlen_q = self.b * self.q_len  # cumulative, as attention_v1 passes it

    def op_kwargs(self, max_seqlen_kv: int, cu=None, seq=None) -> dict:
        return {
            "cu_seqlens_q": self.cu_seqlens_q if cu is None else cu,
            "seqused_kv": self.seqused_kv if seq is None else seq,
            "mask_mode": 3,
            "max_seqlen_q": self.max_seqlen_q,
            "max_seqlen_kv": max_seqlen_kv,
            "layout_q": "TND",
            "layout_q_descale": "TND",
            "layout_kv": "PA_BBND",
            "layout_out": "TND",
        }

    def reseed(self, seed: int) -> None:
        """Rewrite every input in place, the way the next step would.

        Lengths shrink as well as contents, so the refreshed plan differs too --
        otherwise the replay check would only exercise the cache read.
        """
        torch.manual_seed(seed)
        self.k_cache.copy_(torch.randn_like(self.k_cache))
        self.v_cache.copy_(torch.randn_like(self.v_cache))
        self.q.copy_(torch.randn_like(self.q))
        self.kv_lens = [max(1, length - 37) for length in self.kv_lens]
        self.seqused_kv.copy_(torch.tensor(self.kv_lens, dtype=torch.int32))

    def metadata(self, max_seqlen_kv: int, v_descale=None) -> torch.Tensor:
        # Defaults to the placeholder, the way the builder calls it.
        return torch.ops._C_ascend.npu_quant_flash_attn_metadata(
            self.nq,
            self.nkv,
            self.d,
            1,
            v_descale=v_descale_stub() if v_descale is None else v_descale,
            **self.op_kwargs(max_seqlen_kv),
        )


def qfa_call(step: Step, quant, quant_v, block_table, metadata, op_kwargs, mask) -> torch.Tensor:
    """Mirrors AscendAttentionBackendImpl._qfa_paged_call exactly."""
    d = step.d
    q_fp8, q_descale = quant(step.q, d)
    nb, bs = step.k_cache.shape[0], step.k_cache.shape[1]
    k_fp8, k_descale = quant(step.k_cache.reshape(nb, bs, step.nkv, d), d)
    v_fp8, v_descale = quant_v(step.v_cache.reshape(nb, bs, step.nkv, d))
    out, _ = torch.ops._C_ascend.npu_quant_flash_attn(
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
        1,
        block_table=block_table,
        attn_mask=mask,
        metadata=metadata,
        softmax_scale=step.scale,
        **op_kwargs,
    )
    return out


def _exactness(tag: str, got: torch.Tensor, want: torch.Tensor) -> bool:
    if torch.equal(got, want):
        print(f"  [{tag}] bit-exact=True")
        return True
    diff = (got.to(torch.float32) - want.to(torch.float32)).abs()
    print(f"  [{tag}] bit-exact=False max_abs_diff={diff.max().item():.6g} mean={diff.mean().item():.6g}")
    return False


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------
def case_vdescale_stub(args, quant, quant_v) -> bool:
    """The plan is built before any layer quantizes, so it can only pass a
    placeholder v_descale. Prove that costs nothing under PA_BBND.

    The verdict is the *output*, not the plan bytes: the op allocates its
    4096-int32 output with at::empty and writes only a header plus sectionNum
    sections, so the tail is whatever the allocator last left there. The control
    below shows that directly -- two calls with identical arguments already
    disagree on those bytes.
    """
    step = Step(args, args.seed)
    mask = causal_mask()
    max_kv = args.max_seqlen_kv
    _, v_descale = quant_v(step.v_cache.reshape(step.num_blocks, step.block_size, step.nkv, step.d))
    print(f"  real v_descale shape={tuple(v_descale.shape)} vs stub (1, 1, 1, 1, 2)")

    # All three are held at once so each gets its own buffer: released back to
    # back they would reuse one block and inherit the same leftovers, which
    # would make the control agree for the wrong reason.
    plan_real = step.metadata(max_kv, v_descale=v_descale)
    plan_stub = step.metadata(max_kv)
    plan_real2 = step.metadata(max_kv, v_descale=v_descale)
    torch.npu.synchronize()
    a, b, c = plan_real.cpu(), plan_stub.cpu(), plan_real2.cpu()
    control_diff = int((a != c).sum())
    stub_diff = int((a != b).sum())
    print(f"  [control] real vs real, separate buffers: {control_diff} of {a.numel()} int32 differ")
    print(f"  [stub vs real] {stub_diff} of {a.numel()} int32 differ")

    out_real = qfa_call(step, quant, quant_v, step.block_table, plan_real, step.op_kwargs(max_kv), mask)
    out_stub = qfa_call(step, quant, quant_v, step.block_table, plan_stub, step.op_kwargs(max_kv), mask)
    torch.npu.synchronize()
    ok = _exactness("stub-planned vs real-planned output", out_stub.cpu(), out_real.cpu())

    if control_diff:
        print(
            "  plan bytes carry buffer leftovers (identical arguments already disagree), "
            "so only the output is meaningful here"
        )
    elif stub_diff:
        # Deterministic plans that the stub changes: the engine builds its plan
        # before any layer has a real v_descale, so this would have to be fixed
        # by handing the builder a correctly shaped placeholder.
        print("  [RED] plans are deterministic and the stub changed them; v_descale is not inert")
        ok = False
    return ok


def _graph_roundtrip(args, quant, quant_v, max_kv: int, report_memory: bool) -> bool:
    """Capture quant + main op; replay unchanged, then after an in-place swap."""
    import torch_npu

    step = Step(args, args.seed)
    mask = causal_mask()

    md = step.metadata(max_kv)
    print(f"  metadata: numel={md.numel()} dtype={md.dtype} max_seqlen_kv={max_kv}")

    # Eager reference for batch 1.
    ref_a = qfa_call(step, quant, quant_v, step.block_table, md, step.op_kwargs(max_kv), mask).cpu()
    torch.npu.synchronize()

    for _ in range(2):  # warm the allocator/tiling caches before capture
        qfa_call(step, quant, quant_v, step.block_table, md, step.op_kwargs(max_kv), mask)
    torch.npu.synchronize()

    if report_memory:
        torch.npu.reset_peak_memory_stats()
        before = torch.npu.memory_reserved()

    stream = torch_npu.npu.current_stream()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        # Allocated inside the capture, exactly like QFAGraphBuffers.like, and
        # deliberately left unfilled: filling here would record the copies into
        # the graph and replay would read capture-time sources.
        g_cu = torch.empty_like(step.cu_seqlens_q)
        g_seq = torch.empty_like(step.seqused_kv)
        g_md = torch.empty_like(md)
        g_bt = torch.empty_like(step.block_table)
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        out_buf = qfa_call(
            step, quant, quant_v, g_bt, g_md, step.op_kwargs(max_kv, cu=g_cu, seq=g_seq), mask
        )

    if report_memory:
        # No synchronize here: the captured event wait is still outstanding
        # until release() below, so the device would never come back. Reserved
        # bytes are host-side allocator bookkeeping anyway.
        grew = torch.npu.memory_reserved() - before
        cache_bytes = step.k_cache.numel() * 2 * 2  # K + V, bf16
        print(
            f"  [pool] reserved +{grew / 2**20:.1f} MiB capturing one layer over "
            f"{step.num_blocks} blocks ({cache_bytes / 2**20:.1f} MiB of bf16 K+V)"
        )
        print(
            f"  [pool] ratio {grew / cache_bytes:.2f}x the layer's bf16 KV cache -- multiply by the "
            "real per-layer cache to size the graph pool"
        )

    update_stream = torch_npu.npu.Stream()

    def release(source_md: torch.Tensor) -> None:
        """Mirrors _update_qfa_graph_buffers: contents only, then the event."""
        with torch.npu.stream(update_stream):
            g_cu.copy_(step.cu_seqlens_q)
            g_seq.copy_(step.seqused_kv)
            g_md.copy_(source_md)
            g_bt.copy_(step.block_table)
            event.record(update_stream)

    release(md)
    graph.replay()
    torch.npu.synchronize()
    ok = _exactness("A replay(same inputs) vs eager", out_buf.cpu(), ref_a)

    # B: a genuinely different step. The cache is rewritten in place, so a
    # capture that baked the quantized bytes instead of re-reading would show up
    # here and nowhere else.
    step.reseed(args.seed + 1)
    md_b = step.metadata(max_kv)
    ref_b = qfa_call(step, quant, quant_v, step.block_table, md_b, step.op_kwargs(max_kv), mask).cpu()
    torch.npu.synchronize()
    release(md_b)
    graph.replay()
    torch.npu.synchronize()
    ok &= _exactness("B replay(new cache contents) vs eager", out_buf.cpu(), ref_b)

    if torch.equal(ref_a, ref_b):
        print("  [WARN] the two batches produced identical eager output; B proves nothing")
        ok = False
    return ok


def case_quant_in_graph(args, quant, quant_v) -> bool:
    return _graph_roundtrip(args, quant, quant_v, args.max_seqlen_kv, report_memory=False)


def case_maxkv(args, quant, quant_v) -> bool:
    return _graph_roundtrip(args, quant, quant_v, args.max_model_len, report_memory=False)


def case_pool_mem(args, quant, quant_v) -> bool:
    _graph_roundtrip(args, quant, quant_v, args.max_seqlen_kv, report_memory=True)
    print("  [POOL-MEM] reports only; read the numbers above")
    return True


RUNNERS = {
    "VDESCALE-STUB": case_vdescale_stub,
    "QUANT-IN-GRAPH": case_quant_in_graph,
    "MAXKV": case_maxkv,
    "POOL-MEM": case_pool_mem,
}


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cases", default=",".join(CASES), help="comma-separated subset")
    parser.add_argument("--case", help=argparse.SUPPRESS)  # child process entry
    parser.add_argument("--num-heads", type=int, default=24)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--kv-lens", default="300,257,128,65")
    parser.add_argument("--q-len", type=int, default=4, help="tokens per request (1 + MTP drafts)")
    parser.add_argument("--blocks", type=int, default=64, help="cache blocks; raise for POOL-MEM")
    parser.add_argument("--max-seqlen-kv", type=int, default=2048, help="capture constant")
    parser.add_argument("--max-model-len", type=int, default=133120, help="MAXKV's capture constant")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention-v1", help="path to attention_v1.py (default: the installed one)")
    parser.add_argument("--case-timeout", type=float, default=300.0)
    return parser


def run_child(args) -> int:
    print(f"== {args.case} ==")
    try:
        quant, quant_v = bootstrap(args)
        ok = RUNNERS[args.case](args, quant, quant_v)
    except Exception as exc:
        import traceback

        print(f"  [error] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        ok = False
    print(f"  [{args.case}] {'GREEN' if ok else 'RED'}")
    return 0 if ok else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.case:
        return run_child(args)

    names = [n.strip() for n in args.cases.split(",") if n.strip()]
    unknown = [n for n in names if n not in RUNNERS]
    if unknown:
        print(f"[RED] unknown case(s) {unknown}; known: {list(RUNNERS)}")
        return 2

    passthrough: list[str] = []
    skip = False
    for arg in sys.argv[1:]:
        if skip:
            skip = False
        elif arg == "--cases":
            skip = True
        elif not arg.startswith("--cases="):
            passthrough.append(arg)

    results: dict[str, str] = {}
    for name in names:
        command = [sys.executable, os.path.abspath(__file__), "--case", name, *passthrough]
        try:
            completed = subprocess.run(command, timeout=args.case_timeout)
            results[name] = "GREEN" if completed.returncode == 0 else "RED"
        except subprocess.TimeoutExpired:
            print(f"  [{name}] TIMEOUT after {args.case_timeout}s -- likely a stalled event handshake")
            results[name] = "TIMEOUT"

    print("\n=== summary ===")
    for name in names:
        print(f"  {name:<16} {results[name]}")
    return 0 if all(v == "GREEN" for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
