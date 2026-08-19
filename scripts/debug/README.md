# Qwen3.8-2.4T ModelSlim startup diagnosis

Run this inside the serving container on **one node**. It does not start vLLM,
load weights, or allocate NPU memory.

```bash
cd /home/hajimi/qwen3.8
bash scripts/debug/collect_qwen38_modelslim_debug.sh \
  /mnt/share/weight/Qwen3.8-2.4T-A95B-w8a8 \
  /home/hajimi/vllm-ascend
```

If the checkout uses a different path, replace the second argument. It is also
optional: the collector checks `/opt/vllm-ascend`,
`/home/hajimi/vllm-ascend`, and `/home/hajimi/vLLm-ascend` automatically.

The command exits `1` for a reproducible metadata/source mismatch and `0` when
the static contract passes. Send back the path printed as `REPORT_SAVED` and
its contents. The collector only prints a small allowlist of environment
variables; it does not print proxy variables, tokens, or the full environment.

To test the checker itself without model weights:

```bash
python3 scripts/debug/check_qwen38_modelslim_metadata.py --self-test
```
