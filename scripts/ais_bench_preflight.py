#!/usr/bin/env python3
"""ais_bench 开跑前替它先发一个请求，把服务端拒绝的原因打出来。

ais_bench 对失败的请求只留 HTTP 状态短语（base_api.py 里
output.error_info = response.reason），响应体直接丢掉。而 vLLM 把「为什么拒」写在
响应体里，于是上下文超限、缺 chat template、采样参数不支持……到了 ais_bench 这边
统统只剩 "Bad Request" 两个词。接着 warmup 全败，它照样往下跑评测阶段，再报一个
不相干的错，最后汇总出一张空表，退出码还是 0。

这里照 ais_bench 的拼法造出同一个请求体先发一次，一轮答完四件事：

  1. 这个 URL 走不走代理。ais_bench 的 aiohttp 开着 trust_env，容器的 bashrc 又会
     source 代理脚本，no_proxy 里没有目标主机时 localhost 的请求也会被送去代理。
  2. GET  v1/models            谁在应答、它的 max_model_len 是多少。容器是
     --net=host，宿主机上任何服务都可能占着模板里那个端口。
  3. POST v1/chat/completions  非 200 时打印状态码和响应体。
  4. 数据集自身的体检，目前只有 gsm8k 的参考答案格式。

请求体里的 max_tokens / generation_kwargs / model 取自 ais_bench 实际要加载的那份
模型配置，不用默认值顶替：调大的 max_out_len 正是撑爆上下文的常见原因。新版
ais_bench 还能在命令行上覆盖这些字段，所以 `--` 之后接收透传给 ais_bench 的那串
参数，从里面认出会改变请求的几个，按它的规则（只覆盖配置里已有的 key、整体替换）
套上去；其余参数一概不看。

第 3 步固定以 stream=True 发送，收到第一个数据块就断开。拒绝请求的校验（渲染
chat template、长度检查、采样参数检查）都发生在生成开始之前，这样预检只花一次
prefill 的时间，max_out_len 再大也不用等它生成完。代价是要多认一种形态：采样参数
是在生成器里面校验的，流式下那时 200 已经发出去了，vLLM 只能把错误塞进第一个 SSE
事件（data: {"error": ...}）；同一个错误在 ais_bench 的非流式请求那边就是 400。
所以状态码 200 不算数，还要看第一个事件里有没有 error。

单独跑也可以：
    python3 ais_bench_preflight.py --host-ip localhost --host-port 6969 \\
        --template vllm_api_general_chat --dataset gsm8k_gen_0_shot_cot_chat_prompt \\
        -- --max-out-len 32768

退出码 0 = GREEN，1 = RED。只用标准库；要在本机装的 ais_bench 里找模板和数据集，
所以得用 ais_bench 自己的那个解释器跑。
"""

import argparse
import ast
import importlib.util
import json
import os
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request

from gen_ais_bench_model_cfg import find_template

PROMPT = "What is 12 * 12? Let's think step by step."
BODY_LIMIT = 2000
# 需要原样进请求体的字段。它们要是求不出值，就宁可不发，也不猜一个。
NEEDED_FIELDS = ("model", "stream", "api_key", "max_out_len", "generation_kwargs")
# ais_bench 里 VLLMCustomAPIChat.__init__ 的默认值，配置里没写时生效的就是它。
DEFAULT_MAX_OUT_LEN = 4096
_UNPARSED = object()


class Red(Exception):
    pass


def say(message: str) -> None:
    print(f"[preflight] {message}", file=sys.stderr)


def evaluate(node: ast.AST):
    """静态求值 dict(...) 调用和字面量；里面有任何求不出的东西就整体放弃。"""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "dict" and not node.args):
        result = {}
        for keyword in node.keywords:
            value = _UNPARSED if keyword.arg is None else evaluate(keyword.value)
            if value is _UNPARSED:
                return _UNPARSED
            result[keyword.arg] = value
        return result
    try:
        return ast.literal_eval(node)
    except ValueError:
        return _UNPARSED


def load_model_cfg(path: pathlib.Path) -> dict:
    """读 `models = [dict(...)]` 的第一项，不 import、不执行。

    这些配置 import 了 ais_bench 自己的模块，真执行要拖进来一大串依赖；而这里要的
    只是几个字面量。type=SomeClass 这类求不出值的字段直接略过，NEEDED_FIELDS 除外。
    """
    hint = "（设 PREFLIGHT=0 可跳过预检）"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        raise Red(f"读不了模型配置 {path}: {exc}{hint}")

    for stmt in tree.body:
        if not (isinstance(stmt, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "models"
                for target in stmt.targets)):
            continue
        entries = getattr(stmt.value, "elts", None)
        first = entries[0] if entries else None
        if not (isinstance(first, ast.Call) and isinstance(first.func, ast.Name)
                and first.func.id == "dict"):
            break
        cfg = {}
        for keyword in first.keywords:
            if keyword.arg is None:
                raise Red(f"{path} 的 models[0] 用了 ** 展开，没法静态读取{hint}")
            value = evaluate(keyword.value)
            if value is not _UNPARSED:
                cfg[keyword.arg] = value
            elif keyword.arg in NEEDED_FIELDS:
                raise Red(f"{path} 里的 {keyword.arg} 不是字面量，没法静态读取{hint}")
        return cfg
    raise Red(f"{path} 里没找到 models = [dict(...)]，模板格式已变{hint}")


def apply_cli_overrides(args: argparse.Namespace, cfg: dict, passthrough: list) -> None:
    """ais_bench 的 _apply_cli_api_model_overrides 里会改变请求的那一部分。"""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--host-ip")
    parser.add_argument("--host-port", type=int)
    parser.add_argument("--url")
    parser.add_argument("--model-name")
    parser.add_argument("--api-key")
    parser.add_argument("--max-out-len", type=int)
    parser.add_argument("--generation-kwargs", type=json.loads)
    given, _ = parser.parse_known_args(passthrough)

    for name in ("host_ip", "host_port", "url", "model_name"):
        if getattr(given, name) is not None:
            setattr(args, name, getattr(given, name))
    for name in ("api_key", "max_out_len", "generation_kwargs"):
        if getattr(given, name) is not None and name in cfg:
            cfg[name] = getattr(given, name)


def base_url(args: argparse.Namespace) -> str:
    """与 ais_bench 的 BaseAPIModel._get_base_url 同一套规则。"""
    if args.url:
        url = args.url.strip()
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        parsed = urllib.parse.urlparse(url)
        if parsed.path and not parsed.path.endswith("/"):
            # 不补斜杠的话 urljoin 会把最后一段路径吃掉。
            url = urllib.parse.urlunparse(parsed._replace(path=parsed.path + "/"))
        return url
    host = args.host_ip
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{args.host_port}/"


def redact(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if "@" not in parsed.netloc:
        return url
    return urllib.parse.urlunparse(
        parsed._replace(netloc="***@" + parsed.netloc.rsplit("@", 1)[1]))


def resolve_proxy(url: str):
    """aiohttp 在 trust_env=True 下的同一个判定：先看 no_proxy，再按 scheme 取代理。

    判定用的就是 urllib 这两个函数，所以这里直接复用；随后把结论显式交给 opener，
    保证预检的请求和 ais_bench 的请求走的是同一条路。
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname and urllib.request.proxy_bypass(parsed.hostname):
        return None
    return urllib.request.getproxies().get(parsed.scheme)


def fetch(opener, request, timeout: float, first_chunk_only: bool = False):
    """返回 (status, reason, body)。HTTP 错误不抛，响应体才是要看的东西。"""
    try:
        with opener.open(request, timeout=timeout) as response:
            if not first_chunk_only:
                return response.status, response.reason, response.read().decode("utf-8", "replace")
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("data:"):
                    return response.status, response.reason, line
            return response.status, response.reason, ""
    except urllib.error.HTTPError as exc:
        return exc.code, exc.reason, exc.read().decode("utf-8", "replace")


def pointer_for(status: int, body: str, max_out_len: int) -> str:
    """响应体说的是服务端视角，这里补一句该拧本仓的哪个旋钮。"""
    if "maximum context length" in body:
        return (f"请求的 max_tokens={max_out_len} 加上 prompt 超过了服务的上下文："
                "调小模型配置里的 max_out_len，或调大起服务时的 MAX_MODEL_LEN")
    if "chat template" in body:
        return "checkpoint 目录没带 chat template（chat_template.jinja / tokenizer_config.json）"
    if status == 404:
        return "路径或模型名对不上；确认这个端口上应答的确实是 vLLM"
    return ""


def check_endpoint(args: argparse.Namespace, cfg: dict) -> None:
    base = base_url(args)
    proxy = resolve_proxy(base)
    if proxy:
        say(f"代理: {base} 会经 {redact(proxy)} 转发（no_proxy 未命中该主机）")
    else:
        say(f"代理: {base} 直连")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({urllib.parse.urlparse(base).scheme: proxy} if proxy else {}))
    via_proxy = "；请求走了代理，把目标主机加进 no_proxy 再试" if proxy else ""

    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"

    models_url = base + "v1/models"
    try:
        status, reason, body = fetch(
            opener, urllib.request.Request(models_url, headers=headers), timeout=5)
    except OSError as exc:
        raise Red(f"GET {models_url} 连不上: {exc}{via_proxy}")
    if status != 200:
        raise Red(f"GET {models_url} -> {status} {reason}{via_proxy}\n{body[:BODY_LIMIT]}")
    try:
        cards = json.loads(body)["data"]
        served = [card["id"] for card in cards]
    except (ValueError, KeyError, TypeError):
        raise Red(f"GET {models_url} 返回的不是 OpenAI 模型列表，这个端口上不是 vLLM:\n"
                  f"{body[:BODY_LIMIT]}")
    if not served:
        raise Red(f"GET {models_url} 的模型列表是空的")
    for card in cards:
        say(f"应答方: id={card['id']} root={card.get('root')} "
            f"max_model_len={card.get('max_model_len')}")

    # ais_bench 的取法：配置里 model 为空就用 /v1/models 的第一项。
    model = args.model_name or cfg.get("model") or served[0]
    if model not in served:
        raise Red(f"模型名 {model!r} 不在服务的列表 {served} 里，vLLM 会回 404；"
                  "核对 MODEL_NAME 与起服务时的 --served-model-name")

    max_out_len = cfg.get("max_out_len", DEFAULT_MAX_OUT_LEN)
    request_body = {"stream": True, "messages": [{"role": "user", "content": PROMPT}]}
    if cfg.get("stream", False):
        request_body["stream_options"] = {"include_usage": True}
    request_body |= cfg.get("generation_kwargs") or {}
    request_body |= {"max_tokens": max_out_len, "model": model}

    chat_url = urllib.parse.urljoin(base, "v1/chat/completions")
    sent = {key: value for key, value in request_body.items() if key != "messages"}
    say(f"POST {chat_url} {json.dumps(sent, ensure_ascii=False)}")
    try:
        status, reason, body = fetch(
            opener,
            urllib.request.Request(
                chat_url, data=json.dumps(request_body).encode("utf-8"), headers=headers),
            timeout=args.timeout,
            first_chunk_only=True,
        )
    except OSError as exc:
        raise Red(f"POST {chat_url} 没等到应答: {exc}{via_proxy}")
    try:
        event = json.loads(body.removeprefix("data:"))
    except ValueError:
        event = None
    in_stream = status == 200 and isinstance(event, dict) and "error" in event
    if status != 200 or in_stream:
        pointer = pointer_for(status, body, max_out_len)
        where = "（错误在流内，非流式请求拿到的会是 4xx）" if in_stream else ""
        raise Red(f"POST {chat_url} -> {status} {reason}{where}{via_proxy}\n"
                  f"服务端响应体（ais_bench 不会显示这一段）:\n{body[:BODY_LIMIT]}"
                  + (f"\n=> {pointer}" if pointer else ""))
    say(f"POST {chat_url} -> 200")


def check_gsm8k_references() -> None:
    """gsm8k_dataset_postprocess 对每条参考答案做 text.split('#### ')[1]，而且发生在
    读任何一条预测之前：有一条不带这个标记，整轮推理跑完后打分阶段直接 IndexError。"""
    spec = importlib.util.find_spec("ais_bench")
    if spec is None or not spec.origin:
        raise Red("没找到 ais_bench 包，确认用的是 ais_bench 自己的解释器")
    # 与 ais_bench 的 get_data_path 同一套规则：默认在包的上一级目录下找。
    default_root = pathlib.Path(spec.origin).parent.parent
    root = pathlib.Path(os.environ.get("AIS_BENCH_DATASETS_CACHE", default_root))
    test_file = root / "ais_bench" / "datasets" / "gsm8k" / "test.jsonl"
    if not test_file.is_file():
        raise Red(f"gsm8k 数据集不在 {test_file}；按 ais_bench 的 README 下载 gsm8k.zip 解压到该目录")

    total = 0
    bad = []
    with test_file.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            answer = json.loads(line).get("answer")
            if not isinstance(answer, str) or "#### " not in answer:
                bad.append((number, answer))
    if bad:
        number, answer = bad[0]
        raise Red(f"{test_file} 有 {len(bad)}/{total} 条参考答案不含 '#### '，打分阶段会 "
                  f"IndexError；第 {number} 行的 answer 是 {str(answer)[-80:]!r}。"
                  "这不是原版 GSM8K 的格式，重新下载 gsm8k.zip")
    say(f"gsm8k 参考答案 {total} 条，格式正常")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host-ip", default="localhost")
    parser.add_argument("--host-port", type=int, default=6969)
    parser.add_argument("--url", help="给了它就忽略 --host-ip / --host-port")
    parser.add_argument("--model-name")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="ais_bench 要加载的模型配置文件")
    source.add_argument("--template", help="ais_bench 自带的模板名，到装好的包里去找")
    parser.add_argument("--dataset", default="", help="数据集配置名，用来决定跑哪些数据集体检")
    parser.add_argument("--timeout", type=float, default=120, help="等首个数据块的秒数")
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        split = argv.index("--")
        argv, passthrough = argv[:split], argv[split + 1:]
    args = parser.parse_args(argv)

    try:
        cfg_path = pathlib.Path(args.config) if args.config else find_template(args.template)
        say(f"模型配置: {cfg_path}")
        cfg = load_model_cfg(cfg_path)
        apply_cli_overrides(args, cfg, passthrough)
        check_endpoint(args, cfg)
        if args.dataset.startswith("gsm8k"):
            check_gsm8k_references()
    except Red as exc:
        say(f"RED: {exc}")
        return 1
    say("GREEN: 服务接受 ais_bench 的请求")
    return 0


if __name__ == "__main__":
    sys.exit(main())
