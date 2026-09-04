#!/usr/bin/env python3

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

    def test_397b_defaults_to_tp8_with_expert_parallelism(self):
        text = (SCRIPTS_DIR / "397B.sh").read_text(encoding="utf-8")

        self.assertIn(
            'MODEL_PATH="${MODEL_PATH:-/mnt/share/weight/qwen3.5-397b-w4a4_multi}"',
            text,
        )
        self.assertIn('TP_SIZE="${TP_SIZE:-8}"', text)
        self.assertIn('--tensor-parallel-size "$TP_SIZE"', text)
        # EP defaults on; the flag has to be conditional so EP=0 can serve as a
        # comparison point rather than needing a second script.
        self.assertIn('if [[ "${EP:-1}" == "1" ]]; then', text)
        self.assertIn("--enable-expert-parallel", text)
        self.assertIn("--quantization ascend", text)
        self.assertIn("quant_model_description.json", text)

    def test_397b_shares_the_27b_switch_vocabulary(self):
        # The two launchers are read side by side during a QFA experiment, so a
        # switch that means one thing here and another there is a trap.
        text = (SCRIPTS_DIR / "397B.sh").read_text(encoding="utf-8")

        self.assertIn("VLLM_ASCEND_ENABLE_QFA=1", text)
        self.assertIn("VLLM_ASCEND_DISABLE_C8_MXFP=1", text)
        self.assertIn("decode_query_len=$((MTP + 1))", text)
        self.assertIn("$((n * decode_query_len))", text)
        # Unlike 27B this one starts with speculative decoding off: the
        # checkpoint is still being brought up.
        self.assertIn('MTP="${MTP:-0}"', text)

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
