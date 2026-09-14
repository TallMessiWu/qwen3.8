#!/usr/bin/env python3
"""把 ais_bench 自带的模型配置改写成指向某个服务的一份新配置。

老版本 ais_bench（早于 2026-08-27 的 #492）没有 --host-ip / --host-port 这组
命令行覆盖，而配置文件本身又读不了环境变量：那些文件 import 了 ais_bench 自己的
模块，mmengine 因此以 lazy import 解析，文件里的调用不会真的执行，写
os.environ.get(...) 只会得到 LazyObject，一调用就 RuntimeError（TMAN-CFG-001）。

所以这里在运行时生成：读 ais_bench 装在本机的那份模板，把地址字段替换成实际值，
写到自己的 config 目录下。不碰 site-packages，也不依赖 ais_bench 的版本。
每个端点一份文件，并发跑多个服务不会互相覆盖。

单独跑也可以，打印生成的文件路径：
    python3 gen_ais_bench_model_cfg.py --out-dir ./.ais_bench_configs \\
        --name qwen38_localhost_6969 --host-ip localhost --host-port 6969
"""

import argparse
import importlib.util
import os
import pathlib
import re
import sys


def find_template(template: str) -> pathlib.Path:
    spec = importlib.util.find_spec("ais_bench")
    if spec is None or not spec.origin:
        sys.exit("RED: 没找到 ais_bench 包，确认它装在当前 python 环境里")
    root = pathlib.Path(spec.origin).parent
    name = template[:-3] if template.endswith(".py") else template
    matches = sorted(root.glob(f"benchmark/configs/models/**/{name}.py"))
    if not matches:
        sys.exit(f"RED: 在 {root}/benchmark/configs/models 下没找到模板 {name}.py")
    return matches[0]


def replace_field(text: str, field: str, literal: str) -> str:
    """把 `field=<旧值>,` 换成 `field=<新值>,`，要求恰好命中一处。"""
    pattern = re.compile(rf"^(?P<indent>\s*){field}\s*=\s*[^,\n]*(?P<comma>,?)\s*$", re.M)
    hits = len(pattern.findall(text))
    if hits != 1:
        sys.exit(f"RED: 模板里 {field} 命中 {hits} 次，预期 1 次，模板格式已变")
    return pattern.sub(lambda m: f"{m.group('indent')}{field}={literal}{m.group('comma')}", text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True, help="config 目录，文件写到它的 models/ 下")
    parser.add_argument("--name", required=True, help="生成的配置名，不带 .py")
    parser.add_argument("--template", default="vllm_api_general_chat", help="ais_bench 自带的模板名")
    parser.add_argument("--host-ip")
    parser.add_argument("--host-port", type=int)
    parser.add_argument("--url")
    parser.add_argument("--model-name")
    args = parser.parse_args()

    src = find_template(args.template)
    text = src.read_text(encoding="utf-8")

    if args.url:
        # url 非空时 ais_bench 会忽略 host_ip/host_port，两者互斥。
        text = replace_field(text, "url", repr(args.url))
    else:
        if args.host_ip is not None:
            text = replace_field(text, "host_ip", repr(args.host_ip))
        if args.host_port is not None:
            text = replace_field(text, "host_port", str(args.host_port))
    if args.model_name:
        text = replace_field(text, "model", repr(args.model_name))

    header = (
        f"# 由 {pathlib.Path(__file__).name} 生成，勿手改；改地址请重跑评测脚本。\n"
        f"# 模板来自 {src}\n"
    )
    out_dir = pathlib.Path(args.out_dir) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.name}.py"
    # 先写临时文件再 rename，同一端点上并发跑两个数据集时不会读到写了一半的配置。
    tmp = out_dir / f".{args.name}.{os.getpid()}.tmp"
    tmp.write_text(header + text, encoding="utf-8", newline="\n")
    tmp.replace(out)
    print(out)


if __name__ == "__main__":
    main()
