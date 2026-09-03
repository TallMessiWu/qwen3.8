# Scripts

Everything here runs on the server unless it says otherwise. Nothing in this
repository serves the model by itself -- the plugin-side adaptation lives in
`vllm-ascend`, and these are the assets used to launch, diagnose and regress it.

## Service launchers

- `27B.sh` -- single-node 8-NPU Qwen3.8-27B-MXFP8 baseline, the current
  workhorse. Cleans the devices through `npu-cleaner.sh`, applies the host
  tuning, and exposes `QFA` / `GRAPH` / `MTP` / `THINKING` / `GPU_MEM_UTIL` /
  `MAX_MODEL_LEN` / `MAX_NUM_SEQS`. `MTP` is `num_speculative_tokens` (3 by
  default, 0 disables speculative decoding), not an on/off flag. The aclgraph
  capture sizes follow it and `MAX_NUM_SEQS` together -- one size per request
  count, stepping by `MTP + 1` -- because a decode batch wider than the largest
  captured size runs eager; that is `MAX_NUM_SEQS` graphs, so the batch bound
  is also the capture-time and pool-memory bound. `CAPTURE_SIZES` still
  overrides the list outright. `THINKING` writes
  `--default-chat-template-kwargs '{"enable_thinking": ...}'`, on by default;
  a request that carries its own `chat_template_kwargs` still overrides it.
  Both launchers carry `--profiler-config` for the torch/NPU profiler, armed
  but idle until `/start_profile` is posted; traces land in `./profiling`
  relative to wherever the launcher was started.
- `serve_qwen3.8_2.4t_4node.sh` -- the four-node 32-NPU launcher for the
  ModelSlim mxfp8 Qwen3.8-2.4T-A95B checkpoint. `2.4T-0.sh` through `2.4T-3.sh`
  are per-machine wrappers that only pin `NODE_RANK`, the IPs and the NIC.
- `serve_qwen3.8_2.4t_single_node_4layer.sh` -- four-layer weight smoke test on
  one node. Refuses a quant description whose entries are all FLOAT, i.e. a
  checkpoint that was never actually quantized.
- `curl.sh` -- serves the test images over a local `http.server` and sends two
  multimodal chat requests, so the payload stays a URL instead of base64.
- `npu-cleaner.sh` -- kills leftover processes on the given device ids.

## Container and install

- `create-container.sh` -- creates the privileged A5 serving container on the
  host and installs the host-mounted vLLM-Ascend checkout into it.
- `install-vllm-ascend.sh` -- the in-container half: editable install from a
  mounted checkout, or a requested package version.
- `debug/pip_install_qfa.sh` -- full editable install with the log captured and
  the real errors extracted from the TBE cascade noise.
- `debug/build_qfa_ops.sh` -- fast-iteration build of the two QFA custom ops
  only, with readable ninja error extraction.
- `debug/diag_qfa_tiling_registry.sh` -- read-only triage for the mass
  "do not registe tiling struct" build failures.

## Checkpoint diagnostics

Pure stdlib, read-only, never import torch/vllm, never touch the NPU. Written
for the 2.4T weight failures; run them before trusting any new weight
directory, whatever its name says.

- `debug/check_quant_desc_qwen35_moe_text.py` -- replays vLLM Ascend's
  load-time packed-module lookup against `quant_model_description.json`.
- `debug/compare_checkpoint_shapes.py` -- diffs tensor names and shapes between
  a quantized checkpoint and its BF16 original, from safetensors headers alone.
- `debug/check_moe_expert_shapes.py` -- explains a MoE `w13` load failure:
  the load branch is chosen from the tensor name, not the shape.
- `debug/verify_expert_split_axis.py` -- proves on the real bytes which axis a
  fused `gate_up` export was split along, by sign-bit correlation.
- `debug/check_chat_template_thinking.py` -- answers "was thinking already on
  before anyone passed `enable_thinking`": renders the checkpoint's chat
  template with the kwarg absent, true and false, and says which pair matches.
  Uses jinja2 when it is importable and falls back to reading the template's
  `enable_thinking` lines when it is not.
- `debug/estimate_hbm_budget.py` -- answers "do N machines have enough HBM":
  splits the checkpoint into EP-sharded, TP-sharded and replicated bytes (only
  the first shrinks when you add nodes), sizes per-token KV and per-request GDN
  state from `config.json`, then sweeps node count and `max_model_len`. A failed
  launch already carries both runtime inputs it needs: pass
  `Loading model weights took X GB` as `--weights-gib` and
  `Available KV cache memory: X GiB` as `--observed-kv-gib`, and the activation
  plus non-torch overhead falls out as the residual -- no NPU time required.

## Device probe (NPU required, negligible memory)

- `debug/probe_npu_memory.py` -- prints the HBM totals torch reports, which is
  the denominator of every budget and is not the number npu-smi shows. Creates
  an ACL context and exits; no weights, no tensors, no collectives.

## QFA operator and graph validation (NPU required)

- `debug/test_junlin_qfa_npu.py` -- numeric smoke test of the vendored
  QuantFlashAttn against a CPU golden ported from the official test assets.
- `debug/run_doc_examples_qfa_npu.py` -- the two official doc call examples
  ported 1:1 onto the vendored API.
- `debug/test_qfa_as_fia_npu.py` -- checks the FIA-call-site swap on device for
  both shapes the call site builds.
- `debug/diag_qfa_metadata_size.py` -- sweeps `num_blocks` across the int32
  plane-offset ceiling that killed the metadata op on a real prefill.
- `debug/test_qfa_fullgraph_repro_npu.py` -- reproduces the full-graph capture
  arrangement outside the engine and sweeps one dimension at a time.

Build logs land in `debug/logs/`, which is gitignored.

## Benchmark

- `bench_qfa_vs_fia.py` -- `--auto` brings `27B.sh` up twice (baseline
  `QFA=0 NO_PREFIX_CACHE=1`, candidate `QFA=1`), measures both and prints the
  table. Keeping the two configurations symmetric is the script's job.

## Runtime helper

- `runtime/qwen38_checkpoint_layer_filter/` skips checkpoint tensors above the
  four-layer smoke-test limit before lazy safetensors loading. The single-node
  four-layer launcher loads it through `PYTHONPATH`.

## Local regression tests

`tests/` holds the checkpoint-filter contract test, the HTTP image payload
regression test, and the service/container default-value tests. They need
neither an NPU nor a running server, so they run on any machine:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s scripts/tests -p 'test_*.py'
```
