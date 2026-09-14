#!/usr/bin/env python3
"""Regression tests for the generated ais_bench model config.

Old ais_bench builds (before the api_model_args group landed on 2026-08-27)
have no --host-ip, so run_ais_bench.sh rewrites the shipped template instead.
That rewrite is a regex over someone else's file, which is the fragile part of
the whole arrangement: if the template ever stops matching, the generator must
fail loudly rather than quietly emit a config still pointing at port 8080.
These tests pin both the happy path and that failure.
"""

import ast
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


GENERATOR = Path(__file__).parents[1] / "gen_ais_bench_model_cfg.py"
TEMPLATE = textwrap.dedent(
    '''\
    from ais_bench.benchmark.models import VLLMCustomAPIChat

    models = [
        dict(
            attr="service",
            type=VLLMCustomAPIChat,
            model="",
            host_ip="localhost",
            host_port=8080,
            url="",
            max_out_len=512,
        )
    ]
    '''
)


def build_fake_package(root: Path, template: str = TEMPLATE) -> Path:
    pkg = root / "ais_bench"
    cfg_dir = pkg / "benchmark" / "configs" / "models" / "vllm_api"
    cfg_dir.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "benchmark" / "__init__.py").write_text("", encoding="utf-8")
    (cfg_dir / "vllm_api_general_chat.py").write_text(template, encoding="utf-8")
    return root


def run_generator(pkg_root: Path, out_dir: Path, *extra):
    return subprocess.run(
        [sys.executable, str(GENERATOR),
         "--out-dir", str(out_dir), "--name", "probe", *extra],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(pkg_root), "PATH": "/usr/bin:/bin"},
    )


class GeneratedModelConfigTest(unittest.TestCase):
    def test_endpoint_fields_are_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_fake_package(Path(tmp) / "pkg")
            out = Path(tmp) / "configs"
            result = run_generator(
                root, out, "--host-ip", "10.0.0.5", "--host-port", "7969"
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            written = Path(result.stdout.strip())
            self.assertTrue(written.is_file())
            text = written.read_text(encoding="utf-8")
            ast.parse(text)
            self.assertIn("host_ip='10.0.0.5'", text)
            self.assertIn("host_port=7969", text)
            # Untouched fields must survive verbatim, including the trailing
            # comma the dict literal needs.
            self.assertIn("max_out_len=512,", text)
            self.assertNotIn("8080", text)

    def test_url_and_model_are_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_fake_package(Path(tmp) / "pkg")
            out = Path(tmp) / "configs"
            result = run_generator(
                root, out, "--url", "http://gw/prefix/", "--model-name", "qwen3.8"
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            text = Path(result.stdout.strip()).read_text(encoding="utf-8")
            ast.parse(text)
            self.assertIn("url='http://gw/prefix/'", text)
            self.assertIn("model='qwen3.8'", text)

    def test_renamed_template_field_fails_loudly(self):
        # A silent miss here would leave the eval pointed at the template's own
        # default port, which looks like a running eval against the wrong box.
        with tempfile.TemporaryDirectory() as tmp:
            drifted = TEMPLATE.replace("host_port=8080,", "service_port=8080,")
            root = build_fake_package(Path(tmp) / "pkg", template=drifted)
            out = Path(tmp) / "configs"
            result = run_generator(
                root, out, "--host-ip", "localhost", "--host-port", "7969"
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("RED", result.stderr)
            self.assertFalse((out / "models").exists())

    def test_missing_package_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            result = run_generator(empty, Path(tmp) / "configs", "--host-port", "7969")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("RED", result.stderr)


if __name__ == "__main__":
    unittest.main()
