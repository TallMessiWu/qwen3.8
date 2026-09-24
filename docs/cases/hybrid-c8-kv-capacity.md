# hybrid C8 的 KV 容量只有 BF16 的四分之一

> **状态**：修复落在本机（`junlin-c8-mxfp-16614`，`2592b07d2` + `a0e305ea4`），**真机待验**。
> **来源**：2026-09 的 Claude Code 会话记忆（`~/.claude/projects/…/memory/`），2026-09-24 迁入本仓。
> `AGENTS.md`「当前状态」有摘要，本文件是根因、验收预期值与 `num_blocks` 反推法。

**现象**（2026-09-21）：同并发下 QFA 的 `block_size * block_num` 约为 FIA 的 1/4，C8 的 KV
占用率高、并发打不上去。上报值 C8 351,436 tokens vs 非 C8 1,121,536 tokens。

**根因**：vLLM `__post_init__` 里 `try_verify_and_update_config()`（0.28.0 第 1061 行）跑在
`quant_config` 赋值（第 1119 行）**之前**，所以 mamba 配置钩子看不到 C8，按 BF16 把 hybrid 的
公共 page 定成「2048 个 BF16 token + conv」；之后平台钩子里的 `refresh_block_size` 再把 block
写死成 512。`get_kv_cache_spec` 把 C8 attention page 垫到 mamba page 那么大，于是 2.4T@TP8 下
2,117,632 B 的 page 只装 270,336 B 有效载荷，87% 是填充。

**修复**：`junlin-c8-mxfp-16614`（从 PR 16614 头部派生）上的 `2592b07d2` + `a0e305ea4`。
block = SSM state 字节数 / (kv_heads × head_dim × 1 byte)，必须是 512 的整数倍，否则报错。
截至 2026-09-21 **只在本机验证过**（单测 + 纯 PyTorch 布局模拟），真机待验。

**真机验收预期**（同配置、同显存预算）：
- 启动日志出现 `Hybrid C8_MXFP KV cache requires block_size=4096 (8 QFA kernel blocks of 512 tokens)`
- `GPU KV cache size` 从 351,436 涨到约 1.87M（max_model_len=26624 时 ×5.3，131K 时 ×7.2）；应当超过 FIA
- 并发下输出不乱码、MTP 接受率不变（布局改了，串扰会表现为高并发下乱码）

**反推技巧**：0.28.0 的上报值是
`int(num_blocks / Σ_group cdiv(单请求最大字节, page 字节) × max_model_len)`，不是 block size 的
整数倍。mamba 组每请求占 1 块（`align` 模式 2 块，再加投机块）。对两个上报值暴力反解
`(max_model_len, mamba 块数, num_blocks)`，得到 FIA=674、C8=726 块——两边 block 数几乎相等，
直接证明 page 字节数相同、差的只是每 page 的 token 数。

**How to apply:** Ascend hybrid 缓冲区是按段排的 `[conv | K,ssm | V]`，不是按 block 排；
attention 视图的首维是 kernel 块数（逻辑块数 × scale）。任何拿调度器 block id 直接索引张量首维的
代码，在 hybrid attention 上都是错的。相关：[`c8-qfa-config-matrix.md`](../reference/c8-qfa-config-matrix.md)、[`qfa-integration.md`](./qfa-integration.md)。
