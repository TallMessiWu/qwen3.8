#!/usr/bin/env python3
"""Regression tests for the ais_bench preflight.

ais_bench keeps only the HTTP reason phrase of a failed request
(``output.error_info = response.reason``), so a server that rejects the eval's
requests surfaces as the two words "Bad Request" and nothing else -- the body,
where vLLM says *why*, is dropped. The preflight exists to send the same request
first and print that body. These tests pin the three things that make it worth
running: the request really is the one ais_bench would send, a rejection's body
reaches the terminal, and the proxy decision matches aiohttp's ``trust_env``.
"""

import json
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PREFLIGHT = Path(__file__).parents[1] / "ais_bench_preflight.py"
WRAPPER = Path(__file__).parents[1] / "run_ais_bench.sh"
MODEL_CFG = textwrap.dedent(
    '''\
    from ais_bench.benchmark.models import VLLMCustomAPIChat
    from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

    models = [
        dict(
            attr="service",
            type=VLLMCustomAPIChat,
            model="",
            stream=False,
            host_ip="localhost",
            host_port=8080,
            url="",
            max_out_len=777,
            generation_kwargs=dict(
                temperature=0.01,
                ignore_eos=False,
            ),
            pred_postprocessor=dict(type=extract_non_reasoning_content),
        )
    ]
    '''
)
CONTEXT_ERROR = {
    "error": {
        "message": "This model's maximum context length is 1024 tokens. "
                   "However, you requested 777 output tokens",
        "type": "BadRequestError",
        "param": "input_tokens",
        "code": 400,
    }
}


class FakeVLLM:
    """A throwaway OpenAI-compatible endpoint on an ephemeral port."""

    def __init__(self, chat_status=200, chat_body=None, model_ids=("qwen3.8",),
                 stream_chunks=1, first_event=None):
        self.posts = []
        self.client_hung_up = threading.Event()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            # Chunked transfer, which is how uvicorn streams SSE, needs 1.1.
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _send(self, status, payload, content_type="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                cards = [
                    {"id": name, "root": "/weights/" + name, "max_model_len": 1024}
                    for name in model_ids
                ]
                self._send(200, json.dumps({"object": "list", "data": cards}).encode())

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                fake.posts.append((self.path, json.loads(self.rfile.read(length))))
                if chat_status != 200:
                    self._send(chat_status, json.dumps(chat_body).encode())
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                chunk = first_event or {"choices": [{"delta": {"content": "2"}}]}
                event = f"data: {json.dumps(chunk)}\n\n".encode()
                try:
                    for _ in range(stream_chunks):
                        self.wfile.write(f"{len(event):X}\r\n".encode() + event + b"\r\n")
                        self.wfile.flush()
                        time.sleep(0.05)
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    fake.client_hung_up.set()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        # shutdown() blocks for up to one poll interval, and the default is 0.5s.
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def run_preflight(*args, env=None):
    base_env = {"PATH": "/usr/bin:/bin"}
    base_env.update(env or {})
    return subprocess.run(
        [sys.executable, str(PREFLIGHT), *args],
        capture_output=True,
        text=True,
        env=base_env,
        timeout=60,
    )


def write_cfg(tmp: str, text: str = MODEL_CFG) -> str:
    path = Path(tmp) / "model_cfg.py"
    path.write_text(text, encoding="utf-8")
    return str(path)


def build_fake_package(root: Path, answers) -> Path:
    """An installed ais_bench reduced to what the preflight looks up in it."""
    pkg = root / "ais_bench"
    cfg_dir = pkg / "benchmark" / "configs" / "models" / "vllm_api"
    cfg_dir.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "benchmark" / "__init__.py").write_text("", encoding="utf-8")
    (cfg_dir / "vllm_api_general_chat.py").write_text(MODEL_CFG, encoding="utf-8")
    data_dir = pkg / "datasets" / "gsm8k"
    data_dir.mkdir(parents=True)
    with (data_dir / "test.jsonl").open("w", encoding="utf-8") as handle:
        for answer in answers:
            handle.write(json.dumps({"question": "q", "answer": answer}) + "\n")
    return root


class PreflightTest(unittest.TestCase):
    def test_request_mirrors_what_ais_bench_would_send(self):
        with tempfile.TemporaryDirectory() as tmp, FakeVLLM() as server:
            result = run_preflight(
                "--host-ip", "127.0.0.1", "--host-port", str(server.port),
                "--config", write_cfg(tmp),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("GREEN", result.stderr)
            # Whoever reads the output has to be able to tell which server
            # answered: with --net=host any box-wide service can sit on a port.
            self.assertIn("/weights/qwen3.8", result.stderr)
            self.assertIn("max_model_len=1024", result.stderr)

            path, body = server.posts[-1]
            self.assertEqual(path, "/v1/chat/completions")
            # The config's own numbers, not defaults: a hand-raised max_out_len
            # is exactly what overflows a server's context.
            self.assertEqual(body["max_tokens"], 777)
            self.assertEqual(body["temperature"], 0.01)
            self.assertIs(body["ignore_eos"], False)
            # model="" means ais_bench takes the first id /v1/models lists.
            self.assertEqual(body["model"], "qwen3.8")
            self.assertEqual(body["messages"][0]["role"], "user")

    def test_cli_overrides_meant_for_ais_bench_reach_the_request(self):
        # Newer ais_bench takes --max-out-len and --generation-kwargs on the
        # command line and they replace the config's values. With thinking on,
        # raising max_out_len is routine, and it is the number that decides
        # whether the request still fits the server's context.
        with tempfile.TemporaryDirectory() as tmp, FakeVLLM() as server:
            result = run_preflight(
                "--host-ip", "127.0.0.1", "--host-port", str(server.port),
                "--config", write_cfg(tmp),
                "--", "--work-dir", "outputs/run1", "--max-out-len", "4321",
                "--generation-kwargs", '{"temperature": 0.6, "top_p": 0.95}', "--debug",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            body = server.posts[-1][1]
            self.assertEqual(body["max_tokens"], 4321)
            self.assertEqual(body["temperature"], 0.6)
            self.assertEqual(body["top_p"], 0.95)
            # Replaced wholesale, as ais_bench does, not merged into the config's.
            self.assertNotIn("ignore_eos", body)

    def test_does_not_wait_for_the_generation_to_finish(self):
        # A rejection happens before the first token, so the first chunk is all
        # the evidence there is. Waiting for the rest would cost a full
        # max_out_len generation -- minutes, once that is raised for thinking.
        with tempfile.TemporaryDirectory() as tmp, \
                FakeVLLM(stream_chunks=200) as server:
            started = time.monotonic()
            result = run_preflight(
                "--host-ip", "127.0.0.1", "--host-port", str(server.port),
                "--config", write_cfg(tmp),
            )
            elapsed = time.monotonic() - started

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertLess(elapsed, 5, "the full stream lasts ten seconds")
            # Hanging up is what makes vLLM abort the request it no longer needs.
            self.assertTrue(server.client_hung_up.wait(timeout=5))

    def test_rejection_body_reaches_the_terminal(self):
        with tempfile.TemporaryDirectory() as tmp, \
                FakeVLLM(chat_status=400, chat_body=CONTEXT_ERROR) as server:
            result = run_preflight(
                "--host-ip", "127.0.0.1", "--host-port", str(server.port),
                "--config", write_cfg(tmp),
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("RED", result.stderr)
            self.assertIn("400", result.stderr)
            self.assertIn("maximum context length is 1024 tokens", result.stderr)
            # The pointer names this repo's knobs, which the body cannot know.
            self.assertIn("MAX_MODEL_LEN", result.stderr)
            self.assertIn("max_out_len", result.stderr)

    def test_rejection_inside_the_stream_is_red_too(self):
        # The preflight streams, and vLLM validates sampling params inside the
        # generator: by then the 200 is already on the wire, so what would have
        # been a 400 for ais_bench's non-streaming request arrives here as an
        # SSE error event. Reading the status alone would call that GREEN.
        rejected = {"error": {"message": "min_p and logit_bias parameters are not "
                                         "yet supported with speculative decoding.",
                              "type": "BadRequestError", "code": 400}}
        with tempfile.TemporaryDirectory() as tmp, \
                FakeVLLM(first_event=rejected) as server:
            result = run_preflight(
                "--host-ip", "127.0.0.1", "--host-port", str(server.port),
                "--config", write_cfg(tmp),
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("RED", result.stderr)
            self.assertIn("not yet supported with speculative decoding", result.stderr)

    def test_model_name_the_server_does_not_serve_is_red(self):
        with tempfile.TemporaryDirectory() as tmp, FakeVLLM() as server:
            result = run_preflight(
                "--host-ip", "127.0.0.1", "--host-port", str(server.port),
                "--config", write_cfg(tmp), "--model-name", "qwen3.8-smoke",
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("RED", result.stderr)
            self.assertIn("qwen3.8-smoke", result.stderr)
            # No point sending a request the server is known to 404.
            self.assertEqual(server.posts, [])

    def test_url_with_a_path_keeps_its_prefix(self):
        with tempfile.TemporaryDirectory() as tmp, FakeVLLM() as server:
            result = run_preflight(
                "--url", f"http://127.0.0.1:{server.port}/prefix",
                "--config", write_cfg(tmp),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            # ais_bench appends a slash before urljoin so the last segment
            # survives; the preflight has to land on the same path.
            self.assertEqual(server.posts[-1][0], "/prefix/v1/chat/completions")

    def test_unreachable_endpoint_is_red(self):
        with FakeVLLM() as server:
            dead_port = server.port
        # The context manager has closed the listener; nothing answers now.
        with tempfile.TemporaryDirectory() as tmp:
            result = run_preflight(
                "--host-ip", "127.0.0.1", "--host-port", str(dead_port),
                "--config", write_cfg(tmp),
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("RED", result.stderr)

    def test_proxy_decision_follows_the_environment(self):
        # aiohttp runs with trust_env=True inside ais_bench, so an http_proxy
        # without a matching no_proxy sends a localhost request to the proxy.
        proxy = "http://user:hunter2@127.0.0.1:1"
        with tempfile.TemporaryDirectory() as tmp, FakeVLLM() as server:
            cfg = write_cfg(tmp)
            endpoint = ("--host-ip", "127.0.0.1", "--host-port", str(server.port))

            bypassed = run_preflight(
                *endpoint, "--config", cfg,
                env={"http_proxy": proxy, "no_proxy": "127.0.0.1"},
            )
            self.assertEqual(bypassed.returncode, 0, bypassed.stderr)

            hijacked = run_preflight(
                *endpoint, "--config", cfg, env={"http_proxy": proxy},
            )
            self.assertEqual(hijacked.returncode, 1)
            self.assertIn("RED", hijacked.stderr)
            self.assertIn("no_proxy", hijacked.stderr)
            self.assertEqual(server.posts[1:], [])

            for result in (bypassed, hijacked):
                self.assertNotIn("hunter2", result.stdout + result.stderr)

    def test_unparseable_config_fails_loudly(self):
        # Guessing a body would turn the preflight into a second opinion about
        # a different request, which is worse than not running it.
        with tempfile.TemporaryDirectory() as tmp, FakeVLLM() as server:
            result = run_preflight(
                "--host-ip", "127.0.0.1", "--host-port", str(server.port),
                "--config", write_cfg(tmp, "models = build_models()\n"),
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("RED", result.stderr)
            self.assertIn("PREFLIGHT=0", result.stderr)
            self.assertEqual(server.posts, [])


class InstalledPackageTest(unittest.TestCase):
    """Paths the preflight resolves inside the installed ais_bench."""

    def test_template_is_read_from_the_installed_package(self):
        with tempfile.TemporaryDirectory() as tmp, FakeVLLM() as server:
            root = build_fake_package(Path(tmp) / "pkg", ["so 2 #### 2"])
            result = run_preflight(
                "--host-ip", "127.0.0.1", "--host-port", str(server.port),
                "--template", "vllm_api_general_chat",
                "--dataset", "gsm8k_gen_0_shot_cot_chat_prompt",
                env={"PYTHONPATH": str(root)},
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(server.posts[-1][1]["max_tokens"], 777)

    def test_gsm8k_references_without_the_marker_are_red(self):
        # gsm8k_dataset_postprocess does text.split('#### ')[1] on every
        # reference before it looks at a single prediction, so one answer
        # without the marker kills scoring after the whole inference has run.
        with tempfile.TemporaryDirectory() as tmp, FakeVLLM() as server:
            root = build_fake_package(Path(tmp) / "pkg", ["so 2 #### 2", "2", "3"])
            result = run_preflight(
                "--host-ip", "127.0.0.1", "--host-port", str(server.port),
                "--template", "vllm_api_general_chat",
                "--dataset", "gsm8k_gen_0_shot_cot_chat_prompt",
                env={"PYTHONPATH": str(root)},
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("RED", result.stderr)
            self.assertIn("#### ", result.stderr)
            self.assertIn("2/3", result.stderr)

    def test_dataset_cache_override_is_honoured(self):
        with tempfile.TemporaryDirectory() as tmp, FakeVLLM() as server:
            root = build_fake_package(Path(tmp) / "pkg", ["2"])
            cache = Path(tmp) / "cache"
            good = cache / "ais_bench" / "datasets" / "gsm8k"
            good.mkdir(parents=True)
            (good / "test.jsonl").write_text(
                json.dumps({"question": "q", "answer": "so 2 #### 2"}) + "\n",
                encoding="utf-8",
            )
            result = run_preflight(
                "--host-ip", "127.0.0.1", "--host-port", str(server.port),
                "--template", "vllm_api_general_chat",
                "--dataset", "gsm8k_gen_0_shot_cot_chat_prompt",
                env={"PYTHONPATH": str(root), "AIS_BENCH_DATASETS_CACHE": str(cache)},
            )

            self.assertEqual(result.returncode, 0, result.stderr)


class WrapperTest(unittest.TestCase):
    """run_ais_bench.sh has to run the preflight before it hands over."""

    def build_sandbox(self, root: Path, new_cli: bool) -> dict:
        build_fake_package(root / "pkg", ["so 2 #### 2"])
        bin_dir = root / "bin"
        bin_dir.mkdir()
        # Stands in for the real CLI: --help decides which of the wrapper's two
        # modes is taken, any other call is the eval itself being launched.
        help_text = "--host-ip --host-port --url --model-name" if new_cli else "--models"
        fake = bin_dir / "ais_bench"
        fake.write_text(
            "#!/usr/bin/env bash\n"
            f'if [[ "${{1:-}}" == "--help" ]]; then echo "{help_text}"; exit 0; fi\n'
            f'printf "%s\\n" "$*" >> "{root}/launched"\n',
            encoding="utf-8",
        )
        fake.chmod(0o755)
        (bin_dir / "python3").symlink_to(sys.executable)
        return {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "PYTHONPATH": str(root / "pkg"),
            "VLLM_IP": "127.0.0.1",
        }

    def run_wrapper(self, env: dict, *extra):
        return subprocess.run(
            ["bash", str(WRAPPER), "gsm8k_gen_0_shot_cot_chat_prompt", "--debug", *extra],
            capture_output=True, text=True, env=env, timeout=60,
        )

    def test_rejection_stops_the_eval_before_it_starts(self):
        for new_cli in (True, False):
            with self.subTest(new_cli=new_cli), tempfile.TemporaryDirectory() as tmp, \
                    FakeVLLM(chat_status=400, chat_body=CONTEXT_ERROR) as server:
                root = Path(tmp)
                env = self.build_sandbox(root, new_cli)
                env["VLLM_PORT"] = str(server.port)
                try:
                    result = self.run_wrapper(env)
                finally:
                    generated = WRAPPER.parent / ".ais_bench_configs" / "models" / (
                        f"qwen38_127_0_0_1_{server.port}.py")
                    generated.unlink(missing_ok=True)

                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("maximum context length is 1024 tokens", result.stderr)
                self.assertFalse((root / "launched").exists())

    def test_green_preflight_hands_over_to_ais_bench(self):
        for new_cli in (True, False):
            with self.subTest(new_cli=new_cli), tempfile.TemporaryDirectory() as tmp, \
                    FakeVLLM() as server:
                root = Path(tmp)
                env = self.build_sandbox(root, new_cli)
                env["VLLM_PORT"] = str(server.port)
                try:
                    result = self.run_wrapper(env)
                finally:
                    generated = WRAPPER.parent / ".ais_bench_configs" / "models" / (
                        f"qwen38_127_0_0_1_{server.port}.py")
                    generated.unlink(missing_ok=True)

                self.assertEqual(result.returncode, 0, result.stderr)
                launched = (root / "launched").read_text(encoding="utf-8")
                self.assertIn("--datasets gsm8k_gen_0_shot_cot_chat_prompt", launched)
                self.assertIn("--debug", launched)
                if new_cli:
                    self.assertIn(f"--host-ip 127.0.0.1 --host-port {server.port}", launched)
                else:
                    self.assertIn(f"--models qwen38_127_0_0_1_{server.port}", launched)

    def test_forwarded_arguments_shape_the_preflight_request(self):
        with tempfile.TemporaryDirectory() as tmp, FakeVLLM() as server:
            root = Path(tmp)
            env = self.build_sandbox(root, new_cli=True)
            env["VLLM_PORT"] = str(server.port)
            result = self.run_wrapper(env, "--max-out-len", "4321")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(server.posts[-1][1]["max_tokens"], 4321)
            self.assertIn("--max-out-len 4321", (root / "launched").read_text("utf-8"))

    def test_preflight_can_be_switched_off(self):
        with tempfile.TemporaryDirectory() as tmp, \
                FakeVLLM(chat_status=400, chat_body=CONTEXT_ERROR) as server:
            root = Path(tmp)
            env = self.build_sandbox(root, new_cli=True)
            env.update(VLLM_PORT=str(server.port), PREFLIGHT="0")
            result = self.run_wrapper(env)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "launched").exists())
            self.assertEqual(server.posts, [])


if __name__ == "__main__":
    unittest.main()
