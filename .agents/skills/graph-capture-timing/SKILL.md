---
name: graph-capture-timing
description: 排查与预防「值在错误的时刻被固定」这一族 bug —— ACL 图捕获只记录不执行、编译区里的 Python 求值被烘死、捕获期与 replay 期地址不同。当现象是「eager 对、图错」「长 prompt 吐 EOS 或整段乱码」「一开 draft 图 MTP 接受率就塌」「某缓冲永远是零」,或要改 / review 任何会进 ACL 图的 attention、MoE、model_runner 代码时使用。
---

# 图捕获的时刻错位

vllm-ascend 上已经踩过两次,是同一个族的两个变体。都不是算法错,是**一个正确的值在错误的时刻被读走或被固定**。

## 三种错位

| 变体 | 机制 | 已发生的症状 |
|---|---|---|
| **编译期烘死** | 编译区里求值的 Python 标量被烘成捕获期 dummy run 的值,之后永不更新 | `_fused_output_is_reduced` 取到 dummy run 的 MC2 值 True,prefill 实走 ALLGATHER,all_reduce 整个没执行 → 长 prompt 只吐一个 EOS |
| **捕获只记录不执行** | 设备操作被记录进图但没跑,同一段里的 Python 语句却真跑了 | `value_scale_cache.copy_(...)` 没跑而 `filled_caches.add(...)` 跑了 → V scale 永远全零 → 反量化为零 → `softmax(qk) @ 0` **恰好**为零 |
| **地址不保证一致** | 捕获期与 replay 期的输入张量不是同一块内存 | MTP proposer 的 `dummy_run` 用 `query_start_loc_group[draft_index]`,真实 `_propose` 用 `self.arange` |

## 为什么 target 没事、draft 先中招

vLLM 的 `_warmup_and_capture` 在真捕获前先跑 `cudagraph_num_of_warmups` 次
`_dummy_run(cudagraph_runtime_mode=NONE, force_attention=True)`——eager 执行但强制带真实
attention metadata,所以 target 的惰性初始化在预热轮就真跑了。

**draft 没有对等保护**:它的 dummy_run 在 `_dummy_run` 内部被调,预热轮 `aclgraph_runtime_mode`
是 NONE,而 proposer 只在 `== FULL` 时才建 multi_steps_attn_metadata ⇒ 预热轮 `attn_metadata is None`
⇒ forward 提前返回,走不到惰性初始化那段。**draft 第一次真正执行它就是捕获本身。**

⇒ 判断一段代码危不危险时,永远按 draft 的时间线推,不是 target 的。

## 扫描

```bash
python scripts/checks/scan_capture_timing.py vllm_ascend
```

在任一 vllm-ascend worktree 里跑。默认只出核心档(惰性初始化守卫、已做标记、身份集合标记、
bool 快照、wait_stream),同函数体内出现 `capturing` / `_EXTRA_CTX` 的标 GUARDED 并默认隐藏。
`--noisy` 加扫前向分配与 D2H 同步(误报多),`--all` 连 GUARDED 一起列,`--strict` 有 REVIEW 就退 1。

纯文本匹配,**REVIEW 不等于 bug,只等于「没人核对过」**。核对时问三件事:

1. 守卫体里有没有真正的设备侧操作(`copy_` / `.to(device)` / `.contiguous()` / 算术)?纯 CPU 逻辑无所谓。
2. 这段在 draft 的时间线上,第一次执行会不会落在捕获期?
3. 被固定的那个值,在运行时会不会变(通信方式、容量、本步 token 数、序列长度)?

## 修的时候

**编译期烘死 → 只能用 custom op。** vLLM 以 `fullgraph=True` 编译
(`vllm/compilation/wrapper.py:150`),那里 graph break 是**报错**不是降级,所以
`torch._dynamo.disable` 走不通。custom op 是 dynamo 黑盒,不算 graph break,每次 eager 调用真实求值;
被捕获的 decode 重放捕获时的决策,而 uniform decode 恒在容量内,是对的。

⚠️ **op 必须写进单独的 `out`,不能就地改输入。** 没有 shared expert 时调用方的 `result` 就是
`fused_output` 本身,就地写会改坏它还持有的张量 → 整段乱码。原始代码是把局部名重新绑到新张量,保持这个契约。

**捕获期标记 → 把标记包进 `if not _EXTRA_CTX.capturing:`**,拷贝本身留在图里(幂等,replay 反复写同一个值)。
参考 `attention_v1.py` 的 `reshape_and_cache`,那里有完整注释。

**所有消费点一起改。** 上游契约常是成对的(要么 combine kernel 规约了 routed、shared 单独补,
要么都没规约、最后对和补一次)。`_fused_output_is_reduced` 喂三个消费点
(`_maybe_reduce_routed_output_before_transform` / `_reduce_shared_output_if_needed` /
`_maybe_reduce_final_output`),**只改最终那处会让 shared 被规约两次,从吐一个 EOS 变成整段乱码**——已实测踩中。
改之前先把所有读这个值的地方列全,再决定改哪些。

**地址不一致 → 要么每次 replay 重绑整个调用(FIA 的 `graph_task_update` 是「整个调用重跑一遍」,
不是「重绑张量地址」),要么读捕获期自己持有的 buffer(QFA 走后者)。** 两条路都行,别混着走。

## 相邻的坑(同样只在图里出现)

- 捕获块内 `torch.zeros` 会把清零本身录成图节点,每次 replay 重跑;跨 replay 复用的 buffer 必须在捕获外建好。
- 捕获期禁止 D2H(`.item()` / `.tolist()` / `.cpu()`)。要打 plan 的日志放到 builder 侧。
- 图已入队并停在事件上时 `update_stream.wait_stream(current_stream)` 会**死锁**;正解是在需要的那一点
  `record_event()` 配 `wait_event()`,只等那一点。
- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 会显著改变图池地址布局。
- `ASCEND_LAUNCH_BLOCKING=1` 与 aclgraph 捕获互斥(不是与某个算子互斥),要用它调试须关图。

## 验证

**本机 `torch.compile` 复现不出真机的折叠**——普通属性、custom op、复刻 `__getattr__` 代理三种写法
dynamo 都重新求值了。本机「没复现」不是证据,只有真机是。

区分「eager 对图错」的干净单变量见 `npu-remote-diagnose`:`SPEC_EAGER=1` 只关 draft 图,
`GRAPH=0` / `PIECEWISE` 同时改了 target,**PIECEWISE 下 draft 根本不进图**,所以
「PIECEWISE 正常」不能用来推断注意力后端有没有问题。

两次真机事故的完整推理链（包括被推翻的假设和当时用错的排除依据）在
[`docs/cases/397b-moe-ep-graph-bug.md`](../../../docs/cases/397b-moe-ep-graph-bug.md) 与
[`docs/cases/mtp-accept-graph.md`](../../../docs/cases/mtp-accept-graph.md)——本技能是结论,
那两篇是「当时为什么这么判断」。
