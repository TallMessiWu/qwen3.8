# 一开 draft 图 MTP 接受率就塌：V scale 缓存在捕获期被误标为已填充

> **状态**：已结案（2026-09-09）。机制与修法已提炼进技能 `graph-capture-timing`。
> **来源**：2026-09 的 Claude Code 会话记忆（`~/.claude/projects/…/memory/`），2026-09-24 迁入本仓。

**已结案 2026-09-09。** 根因在 `vllm_ascend/attention/attention_v1.py`
`AscendC8MXFPAttentionBackendImpl.reshape_and_cache`：

```python
value_scale_cache.copy_(value_scale.view(...))   # 捕获期只记录，不执行
filled_caches.add(value_scale_cache)             # 这行 Python 真跑了
```

V 的静态 per-channel scale 只填一次，用 Python 集合按张量身份记「填过了」。
draft 层第一次走到这段是在 ACL 图捕获期间，拷贝没真正发生而标记被置上，
于是 V 的 scale 缓存永远全零 → V 反量化为零 → `softmax(qk) @ 0` **恰好为零**。
target 不受影响，机制已验证（2026-09-14）：vLLM 的 `_warmup_and_capture` 在真捕获前
先跑 `cudagraph_num_of_warmups` 次 `_dummy_run(cudagraph_runtime_mode=NONE,
force_attention=True)`——eager 执行但强制带真实 attention metadata，所以 copy 真跑了、
标志正确置上，等到捕获时 `if` 已为假、拷贝连记录都不记录。draft 没有对等保护：它的
dummy_run 在 `_dummy_run` 内部被调、`aclgraph_runtime_mode` 跟着 target 走，预热轮是
NONE，而 proposer 只在 `== FULL` 时才建 multi_steps_attn_metadata，于是预热轮 metadata
为空 → `attn_metadata is None` → C8MXFP 的 forward 直接 `return output.fill_(0)`，
**走不到 reshape_and_cache**。draft 第一次真正执行那段就是捕获本身。

**为什么 27B/35B 从没中过这两个 bug**（同日查清）：输出截断那个要求通信方式会变，而
`select_moe_comm_method` 开头两道提前返回——非 MoE 直接 None、没开 EP 则无条件
ALLGATHER——27B.sh 里没有 `--enable-expert-parallel`（EP 三件套是 397B 独有）且 TP 默认 1，
所以通信方式是常量，烘死也无所谓。V scale 那个要求 `AscendC8MXFPAttentionBackendImpl`
被实例化，而它只在 `VLLM_ASCEND_ENABLE_QFA=1` 时选中，脚本里 `QFA` 默认 0。
两者都不是模型大小的问题，是配置组合：用 `QFA=1 MTP=3 FULL_DECODE_ONLY` 起 27B 一样会中。
修复是把 `filled_caches.add` 包进 `if not _EXTRA_CTX.capturing`。

**别和 127 兜底搞混（当初那条归因就栽在这）**：`mxfp_c8.py` 的两处 127
（参数初值、`raw[raw == 0] = 127`）保的是**源参数** `layer.v_cache_scale`，
在建模型时一次性做完，从来没出过问题；坏的是把这个正确源值**搬进
`kv_cache[3]`** 的那次 copy。两者隔着一个阶段，`filled_caches` 标记也只拦搬运、
与兜底无关。讽刺的是作者在 127 那段注释里已经写明
"2^-127 there would zero the channel on dequant"——正是实际现象，
只是闸门修在了上游一站。cache 里存的是 E8M0 原始字节，字节 0 即 2^-127≈5.9e-39，
乘 int8 后在 bf16 下溢，所以实测是干净的 0.0。

修复后接受率 0.7143 → 2.1228，逐位条件接受率 0.657/0.087/0.000 →
0.8947/0.7647/0.7949，与 eager 的 0.895/0.765/0.795 一致；
MTP head 十一个子块指纹两种模式完全相同。`SPEC_EAGER=1` 的临时价值随之消失。

**这个坑的形状值得记住，同类还有活的**：ACL 图捕获**只记录不执行**，
所以任何「在前向里惰性初始化 + 用 Python 标记记住已做」的写法，
只要第一次执行落在捕获期，就会永久留下未初始化的缓冲。
`_prepare_c8_scales`（同文件，用 `hasattr(layer, "_c8_scales_prepared")` 守卫，
里面是 `.to(device)` / `.contiguous()` 这类会分配并拷贝的操作）是同一形状，
当前配置走不到、未动，但 C8 非 MXFP 路径上是潜在的同款问题。

**为什么 step 0 还能对 65%**：MTP head 吃两个输入——刚接受 token 的 embedding，
和 **target 在该位置的最终隐状态**。后者已含全部上下文，所以 attention 全死时
它退化成「target 隐状态之上加一层 MLP」，仍是不错的预测器。
step≥1 才塌，因为那时隐状态输入换成了 draft 自己上一步的输出，
而唯一能注入新上下文的通道（attention 读 KV）是断的，于是收敛到不动点
（反复吐同一个 token）。这也解释了 MTP=1 时只是 0.81 vs 0.95 而不塌。

**定位方法（这次真正起作用的）**：在两侧同时采**同一组设备侧指纹**，
对前向做二分。先 proposer 层（确认 draft 前向内部），
再 MTP head 子块（embed/norm/fc/layer/out，锁定到 decoder layer），
再 attention 内部（q/k/原始输出/投影），最后落到「q、k 逐位相同、输出恰好为零」。
指纹用设备张量 `copy_` 累积、跑完统一读一次，别在前向里直接 log
（会图断裂、捕获期还非法）。

**反面教训**：在此之前有九次「猜某个描述符不对 → 改了跑」，
结果**全部逐位相同**。逐位相同本身就是信息：说明改的东西不在这条路径上。
连续两三次逐位 no-op 之后就该换方法，而不是继续猜下一个描述符。
我当时还把 `_v_scale_filled_caches` 写进了「已排除，别重做」——那是错的，
它就是真因；仅凭「只有一条分支有」不能当作排除依据。

**仍然有效的读数与开关**：

- `vllm:spec_decode_num_accepted_tokens_per_pos_total` 是**生存曲线**不是逐位独立
  接受率（`observe_draft` 对 `range(num_accepted)` 全部 +1）。
  用 `scripts/checks/mtp_accept_rate.py` 取绝对计数，别信日志里的两位小数比率。
- `speculative_config.enforce_eager`（脚本里的 `SPEC_EAGER=1`）是干净的单变量开关：
  只关 draft 图，target 照常 FULL 捕获，且不回灌 target 的 CompilationConfig。
  比 `GRAPH=0` / `PIECEWISE` 干净，后两者同时改了 target。
- PIECEWISE 下 draft **根本不进图**（`has_full_cudagraphs()` 为假），
  所以「PIECEWISE 正常」不能用来推断注意力后端有没有问题。

相关：[`qfa-integration.md`](./qfa-integration.md)、[`c8-qfa-config-matrix.md`](../reference/c8-qfa-config-matrix.md)、
技能 `npu-remote-diagnose` 的「脚本交付」与 `scripts/local/README.md`
