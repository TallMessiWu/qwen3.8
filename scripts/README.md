# Scripts

Everything here runs on the server unless it says otherwise. Nothing in this
repository serves the model by itself -- the plugin-side adaptation lives in
`vllm-ascend`, and these are the assets used to launch, diagnose and regress it.

Directories are split by lifetime, because the two kinds of script age very
differently. `bench/`, `checks/` and `setup/` are meant to be re-run for months:
every new checkpoint, CANN version or vendored-operator rebase is a reason to
run them again. `debug/` is the opposite -- scratch space for whatever is being
chased this week, cleared once that question is answered.

## Service launchers

- `27B.sh` -- single-node 8-NPU launcher, the current workhorse. Cleans the
  devices through `npu-cleaner.sh`, applies the host settings, and exposes the
  switches the QFA work needs: `QFA`, `GRAPH`, `MTP` (a token count, not a
  flag), `C8`, `CAPTURE_SIZES`, `MAX_NUM_SEQS`, `CUDAGRAPH_MODE`. Despite the
  name it serves whatever `MODEL_PATH` points at, 35B included.
- `397B.sh` -- single-node Qwen3.5-397B. A copy of `27B.sh` differing only in
  the checkpoint, `--tensor-parallel-size 8`, expert parallelism (`EP=0` turns
  it off for a diagnostic, `PREFILL_MC2=1` moves the MC2/all-to-all switch-over
  off the captured graph size) and the HCCL buffer sizing EP's all-to-all needs;
  every switch behaves identically, which a test pins by diffing the two.
- `2.4T-{0..3}.sh` -- the four-node 2.4T launchers, one per node.
- `serve_qwen3.8_2.4t_4node.sh`, `serve_qwen3.8_2.4t_single_node_4layer.sh` --
  the underlying serve commands those wrap.
- `curl.sh` -- multimodal smoke request against a running server.
- `npu-cleaner.sh` -- frees devices left busy by a killed run.

## bench/ -- operator accuracy and performance (NPU required)

Long-lived. Re-run these after a CANN upgrade, a vendored-operator rebase, or
any change to the attention call site.

- `test_qfa_op.py` -- does the vendored QuantFlashAttn compute the right
  answer at all? Eight self-contained cases against golden data, covering TND,
  PA_BNBD, PA_BBND and N2TGD layouts plus the MTP and aclgraph shapes. Builds
  its own inputs, so it needs no checkpoint and no server.
- `test_qfa_vs_fia.py` -- how does QFA compare with the FIA baseline?
  Three-way accuracy (QFA / FIA on dequantized input / FIA on bf16), which
  separates the quantization loss from the operator difference, plus `--bench`
  for timings across prefill and decode shapes.
- `replay_qfa_dump.py` -- feeds a dump captured from a live server back into
  the operator and checks it reproduces the recorded output bit-for-bit. Proves
  a dump is self-contained enough to reproduce a problem away from the engine,
  which is what an operator bug report has to ship. Capture the dump by setting
  `VLLM_ASCEND_QFA_DUMP_DIR` (see `vllm_ascend/attention/qfa_dump.py`); it only
  works in eager mode, since a D2H copy inside a graph capture fails with
  EE1016 and no Python runs on replay.

## checks/ -- checkpoint and device inspection

Long-lived, cheap, and read-only. Most need neither an NPU nor a server.

- `c8_mxfp_weight_support.py` -- can this checkpoint serve the C8-MXFP8 KV
  cache, and how good are its V scales? Mirrors the framework's own name lookup,
  so a GREEN means the scales really will be found. Reports the zero-scale
  channel count, and names which KV-cache recipe the checkpoint selected when it
  is not the MXFP8 one. Pure stdlib.
- `compare_checkpoint_shapes.py` -- diff tensor names and shapes between a
  quantized checkpoint and its bf16 original. Reads safetensors headers only.
- `estimate_hbm_budget.py` -- will N nodes hold this checkpoint? Derived from
  safetensors headers, no load.
- `chat_template_thinking.py` -- with no `enable_thinking` kwarg, does the
  checkpoint's chat template leave thinking on?
- `probe_npu_memory.py` -- print the HBM totals torch actually sees (NPU
  required, negligible memory).

## setup/ -- build and install

- `build_qfa_ops.sh` -- rebuild the vendored QFA kernels. Note the installed
  `_cann_ops_custom` is left holding QFA only; do not serve from that state.
- `pip_install_qfa.sh` -- install the built package.
- `diag_qfa_tiling_registry.sh` -- `ldd -r` over the custom-op libraries, for
  the "do not registe tiling struct" class of failure.
- `install-vllm-ascend.sh`, `create-container.sh` -- environment bring-up.

## debug/ -- scratch space

One-off diagnostics for whatever is being investigated right now. Nothing here
is expected to survive: once a question is answered, its script goes away rather
than accumulating. If a script turns out to be worth re-running later, it
belongs in `bench/` or `checks/` instead.

## runtime/ -- loaded by the server

- `runtime/qwen38_checkpoint_layer_filter/` skips checkpoint tensors above the
  four-layer smoke-test limit before lazy safetensors loading. The single-node
  four-layer launcher loads it through `PYTHONPATH`.

## tests/ -- tests for the scripts themselves

`tests/` holds the checkpoint-filter contract test, the HTTP image payload
regression test, and the service/container default-value tests. They need
neither an NPU nor a running server, so they run on any machine:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s scripts/tests -p 'test_*.py'
```

## local/ -- this machine only

`local/` builds and runs the local venv that mirrors the server container, for
patch-target checks and CPU unit tests before anything reaches the server. See
`local/README.md`.
