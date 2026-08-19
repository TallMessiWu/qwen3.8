# Qwen3.8 serving scripts

The single-node Qwen3.8-27B MXFP8 baseline is `27B.sh`.

## Four-node 2.4T service

The machine entry points are `2.4T-0.sh` through `2.4T-3.sh`; all four call
`serve_qwen3.8_2.4t_4node.sh`. Node 0 must start first. `VLLM_PORT` now defaults
to `8000`, so `set -u` no longer aborts when the variable was not exported.
The launcher also fails before spawning workers if the model metadata is
missing, the NIC does not own the configured IP, or the visible-device count
does not equal TP.

MTP remains off while debugging the base model. After a successful base-model
request, enable one speculative token on all four nodes with:

```bash
ENABLE_MTP=1 bash scripts/2.4T-0.sh
```

The current upstream guide validates this model on four Atlas 800 A3 nodes,
each with 16 NPUs: DP4/TP16/EP64. The local scripts use the supplied four-node,
8-NPU layout: DP4/TP8/EP32. That A5/Ascend 950 topology has not been established
by the checked-in Qwen3.8-2.4T guide, so passing metadata preflight is not yet
proof that HBM capacity, graph replay, accuracy, or performance will pass.

## ModelSlim preflight

Run the fast check before restarting all four machines:

```bash
bash scripts/debug/collect_qwen38_modelslim_debug.sh \
  /mnt/share/weight/Qwen3.8-2.4T-A95B-w8a8 \
  /home/hajimi/vllm-ascend
```

See `scripts/debug/README.md` for the output and return-code contract.

## Build the A5 image

Docker build cannot see a bind mount that will exist only after `docker run`.
Pass `/home/hajimi/proxy.sh` as a BuildKit secret instead; it is sourced for
the networked build step and is not copied into the image:

```bash
cd /home/hajimi/qwen3.8
DOCKER_BUILDKIT=1 docker build \
  --network=host \
  --secret id=hajimi_proxy,src=/home/hajimi/proxy.sh \
  -t qwen3.8-vllm-ascend:a5-20260819 .
```

If the build host already has direct network access, omit `--secret`. The
Dockerfile pins vLLM to `0.27.1` and vLLM-Ascend to
`efc2dd9e997fa99483f09f09a5cbc2d101b89c71`. Override the latter only when all
four nodes will use the same tested commit:

```bash
docker build --build-arg VLLM_ASCEND_REF=<commit> ...
```

Create the container with the original device and volume layout by running:

```bash
bash scripts/run_qwen38_a5_container.sh
```

Override `IMAGE` or `CONTAINER_NAME` when needed. The script refuses to replace
an existing container with the same name.

At runtime, keep the original `/home:/home` mount. Root's interactive
`.bashrc` conditionally sources `/home/hajimi/proxy.sh` and changes into
`/home/hajimi/qwen3.8`, which is the robust equivalent of appending the
unconditional command by hand.

The host-side custom-op security settings still must be applied outside the
container; a Dockerfile cannot modify the host NPU driver's persistent state.
