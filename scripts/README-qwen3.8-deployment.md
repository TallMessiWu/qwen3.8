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
Dockerfile pins vLLM to `0.27.1` but does not clone vLLM-Ascend. Before creating
the container, make sure every node has the intended checkout at the mounted
runtime path and that all four nodes report the same commit:

```bash
git -C /home/hajimi/qwen3.8/vllm-ascend/main pull --ff-only
git -C /home/hajimi/qwen3.8/vllm-ascend/main rev-parse HEAD
```

Dockerfile cannot declare host runtime mounts or namespace options. The
companion launcher contains the original device/volume layout together with
`--net=host`, `--pid=host`, and `--privileged=true`. Create the container with:

```bash
bash scripts/create-container.sh
```

Override `IMAGE` or `CONTAINER_NAME` when needed. The script refuses to replace
an existing container with the same name. After `/home` is mounted, it removes
only `csrc/output` and `csrc/build_out`, then installs the editable checkout
from `/home/hajimi/qwen3.8/vllm-ascend/main`. If installation fails, the new
container is left running so its build environment can be inspected.

At runtime, keep the original `/home:/home` mount. Root's interactive
`.bashrc` conditionally sources `/home/hajimi/proxy.sh` and changes into
`/home/hajimi/qwen3.8/scripts`, which is the robust equivalent of appending the
unconditional command by hand.

The host-side custom-op security settings still must be applied outside the
container; a Dockerfile cannot modify the host NPU driver's persistent state.
