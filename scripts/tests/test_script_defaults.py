#!/usr/bin/env python3

import difflib
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).parents[1]
PORT_CONSUMERS = (
    "27B.sh",
    "397B.sh",
    "curl.sh",
    "serve_qwen3.8_2.4t_4node.sh",
    "serve_qwen3.8_2.4t_single_node_4layer.sh",
)
SERVICE_LAUNCHERS = (
    "27B.sh",
    "397B.sh",
    "serve_qwen3.8_2.4t_4node.sh",
    "serve_qwen3.8_2.4t_single_node_4layer.sh",
)
MODEL_NAME_CONSUMERS = SERVICE_LAUNCHERS + ("curl.sh",)
QFA_VLLM_ASCEND_REPO = (
    "/home/hajimi/qwen3.8/vllm-ascend/junlin-qfa"
)


class ScriptDefaultsTest(unittest.TestCase):
    def test_vllm_port_consumers_default_to_6969(self):
        self.assertFalse((SCRIPTS_DIR / "hajimi-port.sh").exists())
        for name in PORT_CONSUMERS:
            with self.subTest(script=name):
                text = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
                self.assertIn('VLLM_PORT="${VLLM_PORT:-6969}"', text)
                self.assertNotIn("hajimi-port.sh", text)

    def test_model_name_consumers_default_to_qwen38(self):
        for name in MODEL_NAME_CONSUMERS:
            with self.subTest(script=name):
                text = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
                self.assertIn('MODEL_NAME="${MODEL_NAME:-qwen3.8}"', text)
                self.assertNotIn("qwen3.8-smoke", text)

    def test_service_launchers_serve_the_shared_model_name(self):
        for name in SERVICE_LAUNCHERS:
            with self.subTest(script=name):
                text = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
                self.assertIn('--served-model-name "$MODEL_NAME"', text)

    def test_four_node_launcher_defaults_to_mxfp8_checkpoint(self):
        launcher = SCRIPTS_DIR / "serve_qwen3.8_2.4t_4node.sh"
        text = launcher.read_text(encoding="utf-8")

        self.assertIn(
            'MODEL_PATH="${MODEL_PATH:-/mnt/share/weight/Qwen3.8-2.4T-A95B-mxfp8}"',
            text,
        )
        self.assertNotIn("Qwen3.8-2.4T-A95B-w8a8", text)
        self.assertIn("--quantization ascend", text)
        self.assertIn("quant_model_description.json", text)
        self.assertIn("--dtype bfloat16", text)

    def test_27b_graph_plan_follows_mtp_and_batch_bound(self):
        text = (SCRIPTS_DIR / "27B.sh").read_text(encoding="utf-8")

        # MTP is a token count, not a flag: a bare MTP=1 has to mean one draft
        # token, and the capture sizes have to be derived rather than pasted,
        # or a raised MAX_NUM_SEQS silently runs its widest batches eager.
        self.assertIn('MTP="${MTP:-3}"', text)
        self.assertIn('"num_speculative_tokens\\":$MTP', text)
        self.assertIn("decode_query_len=$((MTP + 1))", text)
        self.assertIn("$((n * decode_query_len))", text)
        self.assertNotIn("num_speculative_tokens\":3}", text)
        self.assertNotIn("CAPTURE_SIZES:-1,4,8", text)

    def test_397b_differs_from_27b_only_in_model_and_parallelism(self):
        # The two launchers are read side by side during a QFA experiment, so a
        # switch that means one thing here and another there is a trap. Pin
        # that by diffing them: anything beyond the header, the checkpoint and
        # the parallel size has drifted and needs to be deliberate.
        lines_27b = (SCRIPTS_DIR / "27B.sh").read_text(encoding="utf-8").splitlines()
        lines_397b = (SCRIPTS_DIR / "397B.sh").read_text(encoding="utf-8").splitlines()

        diff = [
            line
            for line in difflib.unified_diff(lines_27b, lines_397b, n=0, lineterm="")
            if line[:1] in "+-" and not line.startswith(("+++", "---"))
        ]
        expected = [
            "-# Single-node Qwen3.8-27B-MXFP8 baseline, retaining the user's host tuning and",
            "-# optional npu-cleaner workflow while fixing the empty default device list.",
            "+# Single-node Qwen3.5-397B launcher. Deliberately identical to 27B.sh apart",
            "+# from the checkpoint, TP8 and expert parallelism, so every switch means the",
            "+# same thing in both and the two can be read side by side during an experiment.",
            '-MODEL_PATH="${MODEL_PATH:-/mnt/share/weight/Qwen3.8-27B-mxfp8}"',
            '+MODEL_PATH="${MODEL_PATH:-/mnt/share/weight/qwen3.5-397b-w4a4_multi}"',
            "-    --tensor-parallel-size 1 \\",
            "+    --tensor-parallel-size 8 \\",
            "+    --enable-expert-parallel \\",
        ]
        self.assertEqual(diff, expected)

    def test_container_install_defaults_to_qfa_worktree(self):
        create_container = (SCRIPTS_DIR / "setup" / "create-container.sh").read_text(
            encoding="utf-8"
        )
        installer = (SCRIPTS_DIR / "setup" / "install-vllm-ascend.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            f"VLLM_ASCEND_REPO={QFA_VLLM_ASCEND_REPO}", create_container
        )
        self.assertIn(f"repo={QFA_VLLM_ASCEND_REPO}", installer)

    def test_readme_bootstrap_includes_qfa_worktree(self):
        readme = (SCRIPTS_DIR.parent / "README.md").read_text(encoding="utf-8")

        self.assertIn(
            "VLLM_ASCEND_QFA_BRANCH=junlin-qfa",
            readme,
        )
        self.assertIn('"origin/${VLLM_ASCEND_QFA_BRANCH}"', readme)
        self.assertIn('"${VLLM_ASCEND_QFA}"; do', readme)


if __name__ == "__main__":
    unittest.main()
