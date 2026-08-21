#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


FILTER_DIR = Path(__file__).with_name("qwen38_checkpoint_layer_filter")
sys.path.insert(0, str(FILTER_DIR))

import checkpoint_layer_filter  # noqa: E402


class CheckpointLayerFilterTest(unittest.TestCase):
    def test_only_truncated_language_layers_are_filtered(self):
        self.assertFalse(
            checkpoint_layer_filter.should_skip_for_layer_limit(
                "model.layers.3.mlp.gate.weight", 4
            )
        )
        self.assertTrue(
            checkpoint_layer_filter.should_skip_for_layer_limit(
                "model.layers.4.mlp.gate.weight", 4
            )
        )
        self.assertTrue(
            checkpoint_layer_filter.should_skip_for_layer_limit(
                "model.language_model.layers.10.self_attn.q_proj.weight", 4
            )
        )
        self.assertTrue(
            checkpoint_layer_filter.should_skip_for_layer_limit(
                "layers.10.self_attn.q_proj.weight", 4
            )
        )
        self.assertFalse(
            checkpoint_layer_filter.should_skip_for_layer_limit(
                "model.visual.encoder.layers.10.self_attn.q_proj.weight", 4
            )
        )

    def test_patch_preserves_existing_ep_filter(self):
        calls = []

        def original_filter(name, local_expert_ids):
            calls.append((name, local_expert_ids))
            return name.endswith("remote_expert.weight")

        module = SimpleNamespace(should_skip_weight=original_filter)
        checkpoint_layer_filter.patch_weight_utils(module, 4, announce=False)

        self.assertTrue(
            module.should_skip_weight("model.layers.8.mlp.gate.weight", {0})
        )
        self.assertEqual(calls, [])
        self.assertTrue(module.should_skip_weight("remote_expert.weight", {0}))
        self.assertEqual(calls, [("remote_expert.weight", {0})])

    def test_import_loader_patches_weight_utils_after_module_execution(self):
        class FakeLoader:
            @staticmethod
            def exec_module(module):
                module.should_skip_weight = lambda name, _: name == "ep-skipped"

        module = SimpleNamespace()
        loader = checkpoint_layer_filter._WeightUtilsLoader(FakeLoader(), 4)
        loader.exec_module(module)

        self.assertTrue(module.should_skip_weight("model.layers.10.weight", None))
        self.assertTrue(module.should_skip_weight("ep-skipped", None))
        self.assertFalse(module.should_skip_weight("model.layers.3.weight", None))

    def test_launcher_enables_same_four_layer_limit_as_hf_override(self):
        launcher = Path(__file__).parents[1] / "serve_qwen3.8_2.4t_single_node_4layer.sh"
        text = launcher.read_text(encoding="utf-8")

        self.assertIn("QWEN38_CHECKPOINT_LAYER_LIMIT=4", text)
        self.assertIn("qwen38_checkpoint_layer_filter", text)
        self.assertIn('"num_hidden_layers":4', text)
        self.assertIn("QWEN38_ROPE_DISPATCH=GREEN", text)
        self.assertIn(
            "if isinstance(self.rotary_emb, AscendMRotaryEmbedding):", text
        )


if __name__ == "__main__":
    unittest.main()
