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
        # that by diffing them: anything beyond the header, the checkpoint, the
        # parallel size and EP's HCCL buffers has drifted and needs to be
        # deliberate.
        lines_27b = (SCRIPTS_DIR / "27B.sh").read_text(encoding="utf-8").splitlines()
        lines_397b = (SCRIPTS_DIR / "397B.sh").read_text(encoding="utf-8").splitlines()

        diff = [
            line
            for line in difflib.unified_diff(lines_27b, lines_397b, n=0, lineterm="")
            if line[:1] in "+-" and not line.startswith(("+++", "---"))
        ]
        expected = [
            "-# Single-node Qwen3.8-27B-MXFP8 baseline, retaining the user's host tuning and",
            '-# optional npu-cleaner workflow while fixing the empty default device list.',
            '+# Single-node Qwen3.5-397B launcher. Deliberately identical to 27B.sh apart',
            '+# from the checkpoint, TP8 and expert parallelism, so every switch means the',
            '+# same thing in both and the two can be read side by side during an experiment.',
            '+# EP-only, which is why 27B.sh does not carry these. Expert parallelism moves',
            "+# every token's hidden state between ranks twice per MoE layer, and the default",
            '+# HCCL buffer is not sized for that -- the 2.4T launcher needed these same',
            '+# values. All eight ranks sit in one node here, hence PCIe on and RoCE off.',
            '+export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"',
            '+export HCCL_BUFFSIZE_EP="${HCCL_BUFFSIZE_EP:-2048}"',
            '+export HCCL_INTRA_PCIE_ENABLE="${HCCL_INTRA_PCIE_ENABLE:-1}"',
            '+export HCCL_INTRA_ROCE_ENABLE="${HCCL_INTRA_ROCE_ENABLE:-0}"',
            '-MODEL_PATH="${MODEL_PATH:-/mnt/share/weight/Qwen3.8-27B-mxfp8}"',
            '+# EP=0 drops expert parallelism. It is the only way to take the MoE',
            '+# comm-method choice out of the picture: select_moe_comm_method returns',
            '+# all-gather outright without EP, so the MC2/ALLTOALL switch at',
            '+# mc2_tokens_capacity -- which equals the largest captured graph size -- never',
            '+# fires. A 397B MoE wants EP on for real serving; this is a diagnostic.',
            '+ep_args=(--enable-expert-parallel)',
            '+if [[ "${EP:-1}" == "0" ]]; then',
            '+    ep_args=()',
            '+    echo "expert parallelism disabled (EP=0)." >&2',
            '+fi',
            '+',
            '+# PREFILL_MC2=1 sizes the MC2 buffers from max-num-batched-tokens instead of',
            '+# the largest captured graph size (set_mc2_tokens_capacity is its only reader).',
            '+# That size is where select_moe_comm_method stops using MC2 and falls to',
            '+# all-to-all, and that switch-over is where long prompts start answering with',
            '+# an immediate EOS -- so moving it is how the causal link gets tested. Here it',
            '+# goes from 400 to 4096, clamped by the 512-tokens-per-rank MC2 limit.',
            '+# That size also decides how much HCCL window MoeDistributeDispatch demands, and',
            '+# the defaults above are not enough: its tiling check asks for 4433MB here and',
            '+# fails with EZ1008 during profile_run. Raise HCCL_BUFFSIZE, not the EP one --',
            '+# the message blames HCCL_BUFFSIZE_EP, but taking that from 2048 to 5120 left',
            '+# the reported "actual CCL_BUFFSIZE" at 2048MB, which is 2 x HCCL_BUFFSIZE both',
            '+# times. So HCCL_BUFFSIZE=2560 buys a 5120MB window, and it comes out of KV',
            '+# cache -- drop GPU_MEM_UTIL if the KV budget then goes negative.',
            '+additional_config=\'{"enable_cpu_binding":true\'',
            '+if [[ "${PREFILL_MC2:-0}" == "1" ]]; then',
            '+    additional_config+=\',"enable_prefill_mc2":true\'',
            '+    echo "prefill MC2 on: MC2 capacity sized from max-num-batched-tokens." >&2',
            '+fi',
            "+additional_config+='}'",
            '+',
            '+MODEL_PATH="${MODEL_PATH:-/mnt/share/weight/qwen3.5-397b-w4a4_multi}"',
            '-    --tensor-parallel-size 1 \\',
            '+    --tensor-parallel-size 8 \\',
            '+    "${ep_args[@]}" \\',
            '-    --additional-config \'{"enable_cpu_binding":true}\' \\',
            '+    --additional-config "$additional_config" \\',
        ]
        self.assertEqual(diff, expected)

    def test_c8_switch_refuses_a_branch_that_ignores_it(self):
        # VLLM_ASCEND_DISABLE_C8_MXFP only exists on junlin-qfa-c8switch.
        # Elsewhere exporting it changes nothing and the server starts with the
        # C8 cache still on, so the run looks like a bf16 baseline and is not
        # one. That already cost a debugging round, hence the guard.
        for name in ("27B.sh", "397B.sh"):
            with self.subTest(script=name):
                text = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
                guard = text.split('if [[ "${C8:-1}" == "0" ]]; then', 1)[1]
                guard = guard.split("export VLLM_ASCEND_DISABLE_C8_MXFP=1", 1)[0]
                self.assertIn("find_spec", guard)
                self.assertIn("exit 2", guard)
                self.assertIn("junlin-qfa-c8switch", guard)

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
