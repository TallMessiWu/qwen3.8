#!/usr/bin/env python3
"""Answer one question about a checkpoint: with no ``enable_thinking`` kwarg,
does its chat template leave thinking on?

vLLM's side of the answer is already known and does not need a checkpoint:
``vllm/parser/qwen3.py`` reads ``chat_kwargs.get("enable_thinking", True)``, so
``--reasoning-parser qwen3`` assumes thinking whenever nobody says otherwise.
What that parser sees, though, is decided earlier by the template: the Qwen3
family injects an empty ``<think></think>`` block only when the kwarg is
explicitly false, which makes "absent" and "true" the same prompt. Whether the
27B checkpoint kept that convention is a property of its own files, so it can
only be read here, on the machine holding the weights.

Reads the template and nothing else -- no torch, no vllm, no NPU, no weights.
Renders it three ways when jinja2 is importable (it is, wherever transformers
is), because the rendered prompts are the actual evidence; falls back to
printing the ``enable_thinking`` lines when it is not.

    python3 scripts/debug/check_chat_template_thinking.py \
        /mnt/share/weight/Qwen3.8-27B-mxfp8
"""

import argparse
import json
import sys
from pathlib import Path

THINK_OPEN = "<think>"
MESSAGES = [{"role": "user", "content": "1+1=?"}]


def load_template(model_dir: Path) -> tuple[str, str]:
    """Return (template, where it came from)."""
    sidecar = model_dir / "chat_template.jinja"
    if sidecar.is_file():
        return sidecar.read_text(encoding="utf-8"), str(sidecar)

    config = model_dir / "tokenizer_config.json"
    if not config.is_file():
        raise SystemExit(f"RED: neither {sidecar} nor {config} exists")

    template = json.loads(config.read_text(encoding="utf-8")).get("chat_template")
    if not template:
        raise SystemExit(f"RED: {config} carries no chat_template")
    if isinstance(template, list):
        raise SystemExit(
            f"RED: {config} carries a multi-template list; inspect it by hand"
        )
    return template, f"{config} (chat_template field)"


def render(template: str, **kwargs) -> str:
    from jinja2 import Environment
    from jinja2.exceptions import TemplateError

    def raise_exception(message):
        raise TemplateError(message)

    env = Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(
        messages=MESSAGES, add_generation_prompt=True, **kwargs
    )


def tail(prompt: str, keep: int = 120) -> str:
    return repr(prompt[-keep:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="checkpoint directory")
    args = parser.parse_args()

    template, source = load_template(args.model_dir)
    print(f"template source: {source}")

    mentions = [
        line.strip()
        for line in template.splitlines()
        if "enable_thinking" in line
    ]
    print(f"lines mentioning enable_thinking: {len(mentions)}")
    for line in mentions:
        print(f"    {line}")

    try:
        import jinja2  # noqa: F401
    except ImportError:
        print(
            "\njinja2 not importable, so no rendering. Read the lines above: the"
            "\nQwen3 convention is `enable_thinking is defined and enable_thinking"
            "\nis false`, i.e. absent == thinking on."
        )
        return 0

    try:
        absent = render(template)
        on = render(template, enable_thinking=True)
        off = render(template, enable_thinking=False)
    except Exception as exc:  # jinja raises several unrelated types
        print(
            f"\nRED: the template failed to render: {type(exc).__name__}: {exc}."
            "\nA template vLLM cannot render is a bigger problem than the kwarg;"
            "\nreport this before drawing any conclusion about thinking."
        )
        return 1

    print("\nrendered prompt tails (add_generation_prompt=True):")
    print(f"    no kwarg          {tail(absent)}")
    print(f"    enable_thinking=1 {tail(on)}")
    print(f"    enable_thinking=0 {tail(off)}")

    if not mentions and absent == on == off:
        print(
            "\nAMBER: the template ignores enable_thinking entirely -- thinking is"
            "\nwhatever the model does unprompted, and passing the kwarg is inert"
            "\n(harmless: jinja drops unused variables). Nothing to control here."
        )
        return 0

    if absent == on and absent != off:
        print(
            "\nGREEN: absent renders identically to enable_thinking=true, so the"
            "\nprevious runs without --default-chat-template-kwargs were already"
            "\nthinking. Passing it changes no prompt; it only makes the default"
            "\nexplicit and gives THINKING=0 something to flip."
        )
        return 0

    if absent == off and absent != on:
        print(
            "\nRED: absent renders identically to enable_thinking=false. The runs"
            "\nwithout the flag were NOT thinking -- adding it changes the prompt,"
            "\nand any measurement taken before the change is a different config."
        )
        return 1

    prefilled = THINK_OPEN in absent
    print(
        f"\nAMBER: absent matches neither branch. <think> in the absent render:"
        f" {prefilled}. Compare the three tails above by hand."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
