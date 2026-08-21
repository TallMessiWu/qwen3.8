#!/usr/bin/env python3

import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).parents[1]
PORT_CONSUMERS = (
    "27B.sh",
    "curl.sh",
    "serve_qwen3.8_2.4t_4node.sh",
    "serve_qwen3.8_2.4t_single_node_4layer.sh",
)


class ScriptDefaultsTest(unittest.TestCase):
    def test_vllm_port_consumers_default_to_6969(self):
        self.assertFalse((SCRIPTS_DIR / "hajimi-port.sh").exists())
        for name in PORT_CONSUMERS:
            with self.subTest(script=name):
                text = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
                self.assertIn('VLLM_PORT="${VLLM_PORT:-6969}"', text)
                self.assertNotIn("hajimi-port.sh", text)

    def test_four_node_launcher_defaults_to_original_bf16_checkpoint(self):
        launcher = SCRIPTS_DIR / "serve_qwen3.8_2.4t_4node.sh"
        text = launcher.read_text(encoding="utf-8")

        self.assertIn(
            'MODEL_PATH="${MODEL_PATH:-/mnt/share/weight/Qwen3.8-2.4T-A95B}"',
            text,
        )
        self.assertNotIn("Qwen3.8-2.4T-A95B-w8a8", text)
        self.assertNotIn("--quantization ascend", text)
        self.assertNotIn("quant_model_description.json", text)
        self.assertIn("--dtype bfloat16", text)


if __name__ == "__main__":
    unittest.main()
