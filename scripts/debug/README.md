# Qwen3.8-2.4T ModelSlim startup diagnosis

Run this inside the serving container on **one node**. It does not start vLLM,
load weights, or allocate NPU memory.

```bash
cd /home/hajimi/qwen3.8
bash scripts/debug/collect_qwen38_modelslim_debug.sh \
  /mnt/share/weight/Qwen3.8-2.4T-A95B-w8a8 \
  /home/hajimi/qwen3.8/vllm-ascend/junlin-bugfix-modelslim-qwen35-moe-text
```

If the checkout uses a different path, replace the second argument. It is also
optional: the collector first checks the actually imported `vllm_ascend` path,
then the Qwen3.8 bugfix/main worktrees and legacy `/opt` or `/home` locations.

The report includes:

- installed package versions and actual imported `vllm`/`vllm_ascend` paths;
- the active vLLM-Ascend Git commit and its pinned vLLM commit/release;
- checkpoint model type, RoPE parameters, quantization metadata, and shard counts;
- whether `patch_qwen3_5.py` safely dispatches ordinary RoPE and M-RoPE by runtime type.

The command exits `1` for a reproducible metadata/source mismatch and `0` when
the static contracts pass. Send back the path printed as `REPORT_SAVED` and its
contents. The collector only prints a small allowlist of environment variables;
it does not print proxy variables, tokens, or the full environment. It does not
read tensor payloads from weight shards.

To test the checker itself without model weights:

```bash
python3 scripts/debug/check_qwen38_modelslim_metadata.py --self-test
```
