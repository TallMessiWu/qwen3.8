# Script runtime helpers and tests

The service launchers and their local regression tests share two supporting
directories:

- `runtime/qwen38_checkpoint_layer_filter/` skips checkpoint tensors above the
  four-layer smoke-test limit before lazy safetensors loading. The single-node
  four-layer launcher loads it through `PYTHONPATH`.
- `tests/` contains the checkpoint-filter contract test, the HTTP image payload
  regression test, and service/container default-value tests.

Run all local regression tests without creating Python bytecode caches:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s scripts/tests -p 'test_*.py'
```
