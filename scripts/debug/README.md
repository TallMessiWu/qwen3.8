# Debug assets

This directory only contains diagnostics and regression tests used by the
current launch scripts:

- `qwen38_checkpoint_layer_filter/` skips checkpoint tensors above the
  four-layer smoke-test limit before lazy safetensors loading. It is loaded by
  `serve_qwen3.8_2.4t_single_node_4layer.sh` through `PYTHONPATH`.
- `test_qwen38_checkpoint_layer_filter.py` verifies the layer filter and its
  launcher contract.
- `test_curl_large_image_payload.py` prevents image requests from regressing to
  Base64 command-line payloads.
- `test_script_defaults.py` verifies service, container, port, and checkpoint
  defaults.

Run all local regression tests without creating Python bytecode caches:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s scripts/debug -p 'test_*.py'
```
