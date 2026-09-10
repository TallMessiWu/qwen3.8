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

## Accuracy evals

Run against an already-serving endpoint, so they need a server but no NPU of
their own.

- `gsm8.sh` -- GSM8K, zero-shot chain-of-thought chat prompt.
- `gpqa.sh` -- GPQA, zero-shot chain-of-thought chat prompt.
- `mmmu.sh` -- MMMU, multimodal, so the server has to accept images.
- `run_ais_bench.sh` -- the shared entry point all three of them exec.

Aim them at a service with environment variables, not by copying `ais_bench`'s
model configs:

```bash
VLLM_PORT=7969 ./gsm8.sh                     # another service on this box
VLLM_IP=10.0.0.5 VLLM_PORT=8000 ./gpqa.sh    # a service on another box
VLLM_URL=http://gw.example/prefix/ ./gsm8.sh # a gateway with a path
MODEL_NAME=qwen3.8 ./gsm8.sh                 # else /v1/models gets probed
AIS_MODEL_CFG=vllm_api_stream_chat.py ./gsm8.sh
```

Editing the endpoint into `configs/models/vllm_api/*.py` is exactly what this
avoids, and putting `os.environ.get(...)` in one of those files does not work:
they import `ais_bench` modules, so mmengine parses them in lazy-import mode,
where no call in the file is ever executed. The call returns a `LazyObject`
that raises `RuntimeError` on invocation, surfacing as `TMAN-CFG-001 invalid
syntax`. mmengine's own `{{$ENV:default}}` substitution is skipped on that path
too. `ais_bench` instead has an `api_model_args` group -- `--host-ip`,
`--host-port`, `--url`, `--model-name` among others -- applied after a config
is loaded, overwriting only keys the config already has. The wrappers expand
the variables in the shell and pass those flags, leaving the shipped configs
untouched. `VLLM_URL` and the `VLLM_IP`/`VLLM_PORT` pair are mutually
exclusive, because a non-empty `url` makes `ais_bench` ignore host and port.

Extra arguments are forwarded verbatim, so `./gsm8.sh --work-dir ./outputs/run1`
works. `--dump-eval-details` is always on, which is what leaves the per-question
requests and answers under `outputs/` for a wrong answer to be read back.

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
- `test_moe_ep_routing.py` -- does ALLGATHER+EP route every token to the right
  expert? Eight ranks under `torchrun`, no model and no expert weights: feeding
  dispatch's output straight back into combine makes each expert an identity
  map, so after the cross-EP all-reduce the expected output is the input
  itself. A missing or double-claimed entry in `expert_map` breaks that
  equality immediately. Covers QuantType.NONE in eager only, so a GREEN rules
  out "the routing logic is wrong" without vouching for the quantized or
  graph paths.

## checks/ -- checkpoint, device and dump inspection

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
- `mtp_accept_rate.py` -- snapshot `/metrics` around a fixed prompt set and
  report speculative-decoding acceptance as absolute counts plus per-position
  conditional rates. The Prometheus counter is a survival curve, not a
  per-position rate, so read it through this rather than off the log line.
- `msprobe_survey.py` -- are two msprobe dump trees comparable at all? Step
  indices are not: the dumper class is picked off `cudagraph_mode`, and every
  `_dummy_run` burns a step number without writing one, so profile_run and each
  capture warmup shift the graph tree relative to the eager one. Aligns by what
  a step contains instead, and can check that the symptom actually reproduced
  while the dump was being collected.
- `msprobe_first_divergence.py` -- given two comparable trees, which op is the
  first to stop matching? In a single-variable A/B the answer is one line and
  everything after it is downstream noise, which a full graph diff buries.
  Reports ops whose statistics msprobe could not compute first, because an op
  that is invalid in one arm only is already the answer.

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

Currently empty, which is the intended resting state.

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
