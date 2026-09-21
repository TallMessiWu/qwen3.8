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
- `autostart-2.4T.sh` -- the cron entry point for those four. Runs on the
  host, brings the container up if it is down, and launches this machine's
  rank inside it. See below.
- `curl.sh` -- multimodal smoke request against a running server.
- `npu-cleaner.sh` -- frees devices left busy by a killed run.

### Scheduling the four-node service

`autostart-2.4T.sh` is what the scheduler calls, `at` or cron. All four machines get a byte-identical
copy: the rank comes from the machine's own IPv4 address, matched against the
same `LOCAL_IP` values the launchers carry, so nothing inside the script is
per-machine. `--rank N` overrides that when the address table is wrong or a box
is being tested from elsewhere. Only the schedule differs between machines, and
only so that node 0 goes first.

It is a restart, not a health check. `2.4T-N.sh` calls `npu-cleaner.sh`, which
SIGKILLs everything holding an NPU, so a tick that lands while the service is
healthy kills and relaunches it. That is the intended behaviour -- schedule it
on the cadence the service should be recycled on, not every five minutes. A
`flock` keeps two ticks from overlapping, since a second one would reap the
first mid-load, and ranks 1-3 wait for node 0 to bind the DP handshake port
before starting, because they connect to it rather than the other way round.

`SCRIPTS_DIR` defaults to `/home/hajimi/qwen3.8/scripts`, which is also the
`SHELL_WORKDIR` that `setup/create-container.sh` makes the container's
`~/.bashrc` cd into, so the launch shell is already sitting there. Set it if
this checkout lives somewhere else. The script refuses to start rather than
guessing when the launcher is not readable at that path.

Prove the container plumbing before scheduling anything:

```bash
bash autostart-2.4T.sh --check
```

That dumps what `bash -c` and `bash -ic` each end up with inside the container
and requires them to differ, which is the only way to show `~/.bashrc` is
really being sourced -- `bash -c` never reads it and `bash -lc` reads the
profile files instead, and either would start the server in a stripped
environment that fails much later. It touches no NPU and does not restart
anything.

To fire it a single time, use `at` rather than a crontab entry that has to be
removed afterwards. Node 0 at 01:00, nodes 1 to 3 at 01:10, each on its own
machine, and as root because talking to docker needs it:

```bash
echo 'bash /home/hajimi/qwen3.8/scripts/autostart-2.4T.sh >> /home/hajimi/qwen3.8/scripts/logs/autostart-once.log 2>&1' | sudo at 01:00
```

`atq` lists what is queued and `atrm <id>` drops one. If `atd` is not running
(`systemctl is-active atd` says), a transient systemd timer does the same job
without installing anything:

```bash
sudo systemd-run --on-calendar='2026-09-15 01:00' --unit=qwen38-autostart /home/hajimi/qwen3.8/scripts/autostart-2.4T.sh
```

An absolute timestamp matches once, so the unit fires and then cleans itself up.

For a recurring restart instead, install the entry in **root's** crontab
(`sudo crontab -e`). Node 0 at 01:00:

```cron
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
0 1 * * * /home/hajimi/qwen3.8/scripts/autostart-2.4T.sh >> /home/hajimi/qwen3.8/scripts/logs/autostart-cron.log 2>&1
```

Nodes 1 to 3 at 01:10, identical apart from the minute:

```cron
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
10 1 * * * /home/hajimi/qwen3.8/scripts/autostart-2.4T.sh >> /home/hajimi/qwen3.8/scripts/logs/autostart-cron.log 2>&1
```

Ten minutes is far more head start than node 0 needs -- it binds the handshake
port within a minute of starting, long before the weights are read -- and node 0
then sits waiting for the other three, because DP initialisation only completes
once every rank has joined. One or two minutes would do the same job and get the
service up sooner. The in-script wait stays useful either way: it is what covers
a node 0 that is late rather than early.

Add `@reboot sleep 120 && ...` alongside if the service should also come back
after a power cycle; the sleep gives the docker daemon and the NPU driver time
to come up first. The server's own output goes to a timestamped file per run
under `logs/`, kept for `LOG_KEEP_DAYS` (7) days; the crontab redirect above
only catches this script's own progress lines. Nothing is written under `/tmp`.

Switches worth knowing: `STAGGER_SECONDS` (20) is the flat head start ranks 1-3
give node 0 before probing it, `WAIT_NODE0_SECONDS` (300) caps the probe,
`KILL_STALE` (1) reaps a surviving API-server frontend that `npu-cleaner.sh`
would miss because it holds no NPU, and `CONTAINER_NAME` (`hajimi-vllm`) has to
match `setup/create-container.sh`. The script never runs `docker run`: a
missing container is an error telling you to run `setup/create-container.sh`
once, because `docker run` alone would skip the vLLM-Ascend install.

## Accuracy evals

Run against an already-serving endpoint, so they need a server but no NPU of
their own.

- `gsm8k.sh` -- GSM8K, zero-shot chain-of-thought chat prompt.
- `gpqa.sh` -- GPQA, zero-shot chain-of-thought chat prompt.
- `mmmu.sh` -- MMMU, multimodal, so the server has to accept images.
- `run_ais_bench.sh` -- the shared entry point all three of them exec.
- `gen_ais_bench_model_cfg.py` -- rewrites the endpoint into a copy of
  `ais_bench`'s own model template, for builds too old to take it on the
  command line. `run_ais_bench.sh` calls it; it also runs standalone.

Aim them at a service with environment variables, not by copying `ais_bench`'s
model configs:

```bash
VLLM_PORT=7969 ./gsm8k.sh                     # another service on this box
VLLM_IP=10.0.0.5 VLLM_PORT=8000 ./gpqa.sh    # a service on another box
VLLM_URL=http://gw.example/prefix/ ./gsm8k.sh # a gateway with a path
MODEL_NAME=qwen3.8 ./gsm8k.sh                 # else /v1/models gets probed
AIS_MODEL_CFG=vllm_api_stream_chat ./gsm8k.sh
```

Hand-editing the endpoint into `configs/models/vllm_api/*.py` is what this
avoids, and putting `os.environ.get(...)` in one of those files does not work:
they import `ais_bench` modules, so mmengine parses them in lazy-import mode,
where no call in the file is ever executed. The call returns a `LazyObject`
that raises `RuntimeError` on invocation, surfacing as `TMAN-CFG-001 invalid
syntax`. mmengine's own `{{$ENV:default}}` substitution is skipped on that path
too. `VLLM_URL` and the `VLLM_IP`/`VLLM_PORT` pair are mutually exclusive,
because a non-empty `url` makes `ais_bench` ignore host and port.

How the endpoint actually reaches `ais_bench` depends on its version, and the
entry point probes for it rather than being told:

- From the `api_model_args` group added on 2026-08-27 (tag
  `v3.1-20260827-master`), `--host-ip`, `--host-port`, `--url` and
  `--model-name` are applied after a config loads, overwriting only keys it
  already has. The variables expand in the shell into those flags.
- Older builds reject those flags outright. There the generator reads the
  template `ais_bench` installed, rewrites the address fields, and writes the
  result under `.ais_bench_configs/models/`, which `--config-dir` then puts
  ahead of the shipped directory. Datasets still resolve from the shipped one,
  because the lookup takes both and a missing directory is not an error. Each
  endpoint gets its own file, so two services can be evaluated at once, and
  `site-packages` is never touched.

The rewrite is a regex over someone else's file, so it asserts each field
matches exactly once and aborts otherwise: a silent miss would leave the eval
pointed at the template's default port, which reads as a completed run against
the wrong box. `scripts/tests/test_ais_bench_model_cfg.py` pins that.

Extra arguments are forwarded verbatim, so `./gsm8k.sh --work-dir ./outputs/run1`
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
- `packed_shard_quant_uniformity.py` -- would this checkpoint's
  `quant_model_description.json` trip the fused-shard check that
  `get_quant_type_for_layer` runs at model init? Replays it verbatim over every
  fused module the model actually builds, so a RED names the exact prefix and
  whether it raises on a missing shard or on shards disagreeing about their
  quant type. PR #16051 moved `packed_modules_mapping` from a vllm-ascend table
  to vLLM's own model class, so the set of checked groups now follows the
  vendor; `--from-vllm` resolves it from the installed vLLM and says whether the
  table baked into the script has gone stale. Also reports the two things the
  check cannot see: experts past expert 0, and the MTP drafter's unpacked GDN
  projection. Pure stdlib, config plus description only, no safetensors.
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
- `scan_capture_timing.py` -- grep for the code shapes behind the "a value got
  frozen at the wrong moment" bug family: lazy-init guards, done-flags, identity
  set markers, bool snapshots of properties, and `wait_stream`. ACL graph capture
  records without executing, so any of these whose first execution lands inside
  capture leaves a buffer permanently unfilled or a predicate permanently stale;
  both the long-prompt EOS bug and the MTP acceptance collapse were this. Text
  matching only, so false positives are expected -- a REVIEW means nobody has
  checked that line, not that it is wrong. Guard detection is per function body,
  because a window of lines gets fooled by a neighbouring function's guard.
  `--noisy` adds in-forward allocation and D2H sync. Pure stdlib, no NPU.

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
