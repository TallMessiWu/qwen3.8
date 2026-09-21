#!/usr/bin/env python3

import difflib
import re
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
DEFAULT_VLLM_ASCEND_REPO = (
    "/home/hajimi/qwen3.8/vllm-ascend/junlin-c8-mxfp"
)
EVAL_WRAPPERS = {
    "gsm8k.sh": "gsm8k_gen_0_shot_cot_chat_prompt",
    "gpqa.sh": "gpqa_gen_0_shot_cot_chat_prompt",
    "mmmu.sh": "mmmu_gen",
}
EVAL_ENV_VARS = (
    "VLLM_IP",
    "VLLM_PORT",
    "VLLM_URL",
    "MODEL_NAME",
    "AIS_MODEL_CFG",
    "PREFLIGHT",
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
            '+if [[ "${PREFILL_MC2:-0}" == "1" ]]; then',
            '+    additional_config+=\',"enable_prefill_mc2":true\'',
            '+    echo "prefill MC2 on: MC2 capacity sized from max-num-batched-tokens." >&2',
            '+fi',
            '-MODEL_PATH="${MODEL_PATH:-/mnt/share/weight/Qwen3.8-27B-mxfp8}"',
            '+MODEL_PATH="${MODEL_PATH:-/mnt/share/weight/qwen3.5-397b-w4a4_multi}"',
            '-TP="${TP:-1}"',
            '+TP="${TP:-8}"',
            '+    "${ep_args[@]}" \\',
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

    def test_container_install_defaults_to_c8_mxfp_worktree(self):
        create_container = (SCRIPTS_DIR / "setup" / "create-container.sh").read_text(
            encoding="utf-8"
        )
        installer = (SCRIPTS_DIR / "setup" / "install-vllm-ascend.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            f"VLLM_ASCEND_REPO={DEFAULT_VLLM_ASCEND_REPO}", create_container
        )
        self.assertIn(f"repo={DEFAULT_VLLM_ASCEND_REPO}", installer)

    def test_readme_bootstrap_includes_c8_mxfp_worktree(self):
        readme = (SCRIPTS_DIR.parent / "README.md").read_text(encoding="utf-8")

        self.assertIn(
            "VLLM_ASCEND_C8_BRANCH=junlin-c8-mxfp",
            readme,
        )
        self.assertIn('"origin/${VLLM_ASCEND_C8_BRANCH}"', readme)
        self.assertIn('"${VLLM_ASCEND_C8}"; do', readme)

    def test_eval_wrappers_document_every_env_var(self):
        # Whoever opens gpqa.sh should not have to open a second file to learn
        # that VLLM_IP exists. The switches are documented in each wrapper, so
        # pin that the list stays complete wherever it is repeated.
        sources = {name: (SCRIPTS_DIR / name).read_text(encoding="utf-8")
                   for name in EVAL_WRAPPERS}
        sources["run_ais_bench.sh"] = (
            SCRIPTS_DIR / "run_ais_bench.sh"
        ).read_text(encoding="utf-8")

        for name, text in sources.items():
            for var in EVAL_ENV_VARS:
                with self.subTest(script=name, var=var):
                    self.assertIn(var, text)

    def test_gsm8k_wrapper_is_not_shadowed_by_its_old_name(self):
        # The benchmark is GSM8K and the wrapper shipped for a while as gsm8.sh.
        # A leftover copy would keep running, without whatever was fixed since.
        self.assertFalse((SCRIPTS_DIR / "gsm8.sh").exists())

    def test_eval_wrappers_differ_only_in_dataset(self):
        # Three near-identical wrappers drift the way 27B.sh and 397B.sh did.
        # Normalise away the script name and the dataset, then require what is
        # left to be byte-identical, so a fix to one reaches all three.
        normalised = {}
        for name, dataset in EVAL_WRAPPERS.items():
            text = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
            self.assertIn(f'run_ais_bench.sh" {dataset} "$@"', text)
            body = [
                line
                for line in text.splitlines()
                # The opening comment names the benchmark; MMMU also warns that
                # the server has to accept images. Both are meant to differ.
                if not line.startswith("# ") or "精度评测" not in line
            ]
            normalised[name] = "\n".join(body).replace(
                dataset, "<DATASET>").replace(name, "<SCRIPT>")

        reference_name, reference = next(iter(normalised.items()))
        for name, body in normalised.items():
            with self.subTest(script=name):
                self.assertEqual(
                    body,
                    reference,
                    f"{name} has drifted from {reference_name}",
                )


    def test_autostart_agrees_with_the_four_node_launchers(self):
        # autostart-2.4T.sh derives its rank from the machine's own IPv4
        # address so that all four boxes can run a byte-identical copy under an
        # identical crontab line. That only holds while its table agrees with
        # the launchers: an IP changed on one side alone leaves a machine
        # either unable to identify itself or, worse, answering as another
        # rank and loading the wrong half of the DP group.
        autostart = (SCRIPTS_DIR / "autostart-2.4T.sh").read_text(encoding="utf-8")

        for rank in range(4):
            with self.subTest(rank=rank):
                launcher = (SCRIPTS_DIR / f"2.4T-{rank}.sh").read_text(encoding="utf-8")
                local_ip = re.search(r"LOCAL_IP:-([\d.]+)\}", launcher)
                self.assertIsNotNone(local_ip, f"2.4T-{rank}.sh has no LOCAL_IP default")
                self.assertIn(
                    f'NODE{rank}_IP="${{NODE{rank}_IP:-{local_ip.group(1)}}}"',
                    autostart,
                )

        # The rendezvous probe has to watch the port node 0 actually binds.
        serve = (SCRIPTS_DIR / "serve_qwen3.8_2.4t_4node.sh").read_text(encoding="utf-8")
        rpc_port = re.search(r"DP_RPC_PORT:-(\d+)\}", serve)
        self.assertIsNotNone(rpc_port, "the 4-node launcher has no DP_RPC_PORT default")
        self.assertIn(
            f'DP_RPC_PORT="${{DP_RPC_PORT:-{rpc_port.group(1)}}}"', autostart
        )

        create_container = (SCRIPTS_DIR / "setup" / "create-container.sh").read_text(
            encoding="utf-8"
        )
        container_name = re.search(r"CONTAINER_NAME:-([\w.-]+)\}", create_container)
        self.assertIsNotNone(container_name, "create-container.sh has no name default")
        self.assertIn(
            f'CONTAINER_NAME="${{CONTAINER_NAME:-{container_name.group(1)}}}"',
            autostart,
        )

        # The container's ~/.bashrc cds into SHELL_WORKDIR, so launching from
        # anywhere else would run the service out of a different directory than
        # every interactive session that debugs it -- and ./profiling, which the
        # 4-node launcher resolves against the CWD, would land somewhere else too.
        workdir = re.search(r"SHELL_WORKDIR=(\S+)", create_container)
        self.assertIsNotNone(workdir, "create-container.sh has no SHELL_WORKDIR")
        self.assertIn(
            f'SCRIPTS_DIR="${{SCRIPTS_DIR:-{workdir.group(1)}}}"', autostart
        )

    def test_autostart_starts_the_service_from_an_interactive_shell(self):
        # ~/.bashrc is what create-container.sh loads the proxy and the working
        # directory from, and bash reads that file only for interactive shells.
        # `bash -c` skips it silently and `bash -lc` reads the profile files
        # instead; either would start the server in a stripped environment that
        # only fails much later. --detach is the other half of the contract: the
        # cron process exits at once, and the server has to outlive it.
        autostart = (SCRIPTS_DIR / "autostart-2.4T.sh").read_text(encoding="utf-8")

        launch = [
            line
            for line in autostart.splitlines()
            if "docker exec --detach" in line
        ]
        self.assertEqual(len(launch), 1, "expected exactly one detached exec")
        self.assertIn('--user "$CONTAINER_USER"', launch[0])

        body = autostart.split("docker exec --detach", 1)[1]
        self.assertIn("bash -ic \"cd '$SCRIPTS_DIR'", body)
        self.assertNotIn(" -lc ", body)

        # A detached exec discards its own stdout, so the redirect has to run
        # inside the container or the log is silently empty.
        self.assertIn(">>'$log_file' 2>&1", body)


if __name__ == "__main__":
    unittest.main()
