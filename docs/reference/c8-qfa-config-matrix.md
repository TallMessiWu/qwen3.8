# C8 / QFA 可服务配置矩阵

结论：**真正能起服务的只有两种组合。** 开关在 `scripts/27B.sh` / `scripts/397B.sh` 里。

| 组合 | 开关 | KV cache | 注意力算子 | 用途 |
| --- | --- | --- | --- | --- |
| **C16+FIA** | `C8=0` | bf16 | 普通 FIA | 端到端精度 / 显存对比的**唯一**基线 |
| **C8+QFA** | `QFA=1`（默认 C8=1） | MXFP8（`K_DYNAMIC_V_STATIC_MXFP8_PER_CHANNEL`） | QuantFlashAttn 原地读 | 在跑的东西 |

两边都必须 `--no-enable-prefix-caching`（`27B.sh` 里 `C8=0` 或 `QFA=1` 任一为真就自动加）：
MXFP8 cache 的 E8M0 scale 跨 token 共享，共享 block 追踪不了；基线不关的话两次跑的差异
就不止一个算子。

## 「MXFP8 cache + FIA」不是一个配置

head_dim=256 下 FIA 直接 **EZ0010** 挂掉，这正是当初 vendor QFA 的理由。想用它拆
「带宽收益 vs kernel 收益」是行不通的，**别再往 bench 里加这条腿**（加过一次，白写）。
QFA 也读不了 bf16 cache，所以 `C8=0` 蕴含 QFA 关。

## ⚠️ `C8=0` 基线目前拿不到

定义 `VLLM_ASCEND_DISABLE_C8_MXFP` 的 `junlin-qfa-c8switch` 分支已在 2026-09-10 清理时
删除，**本机现存的所有 worktree 里都没有这个环境变量**（含 `main` / `upstream-main` /
`junlin-qfa` / `junlin-c8-mxfp*`，2026-09-24 核对）。`27B.sh` 现在会**明确报错退出 2**
而不是静默哑开关——守护在 `747a883`（2026-09-04）加的，比静默失效好：

```
ERROR: C8=0 needs VLLM_ASCEND_DISABLE_C8_MXFP, which the installed vllm-ascend does not define.
       (...) Serve from the junlin-qfa-c8switch worktree, or drop C8=0.
```

⚠️ 报错文案第二行仍指向那个**已删除的 worktree**，照它做会扑空。要恢复 bf16 基线，
得重新引入这个开关（或改用其它方式关掉 C8 cache），不是换个目录就行。

## QFA 的服务边界

- **`_qfa_serves` 要求 `attn_metadata.qfa is not None`。**
- **PrefillNoCache 也走 QFA。** `_attach_qfa_inputs` 只被 `if self.enable_qfa` 门控，
  函数体里没有任何 state 早返（从 `40f92f52e` 引入时就没有），所以每个 state 都拿到 plan。
  ⚠️ `attention_v1.py` 里那两条说「PrefillNoCache 在顶上就 return 了」/「qfa 在
  PrefillNoCache 下是 None」的注释**是写下时就过期的**，别照着它推。实际后果：长 prompt
  的 prefill 是 QFA 读分页 MXFP8 cache。
- `_get_fia_params` 给 FIA 的同样是 `block_table`，所以两边 prefill 形状一致
  ⇒ `scripts/bench/test_qfa_vs_fia.py --bench` 因此没有 dense 行。
- **`_qfa_paged_call` 只量化 q**；K/V 在 cache 里已经是 FP8，且已是 PA_BBND 要的顺序，
  读侧不量化也不 transpose。量化只发生在写侧（`reshape_and_cache` 之前）。

## 与 block size 的交叉点

QFA 读的 **kernel 块是 512**；调度用的逻辑块在 hybrid 模型上不该是 512——这就是
hybrid C8 KV 容量塌成四分之一的那条线，详见 `../cases/hybrid-c8-kv-capacity.md`。

## 上游路线对照

`FullAttentionSpec.head_size` 撑成 `head_size + head_size//32` 骗过 vLLM 的块预算、
再把 raw buffer 切成 k / v / k_scale / v_scale 四份的那套框架侧做法，来自同事的
`Tame21/vllm-ascend` 分支 `cache-c8`；他们那侧 FIA 图路径**不支持 MTP**
（`_full_graph_mxfp8_decode` 遇到 `len(actual_seq_qlen) != num_tokens` 就抛），
我们的 QFA 捕获两条都不设限。完整来龙去脉见 `../cases/qfa-integration.md`。
