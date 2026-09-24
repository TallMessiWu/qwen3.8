# 397B 长 prompt 只吐一个 EOS：MoE 规约判据在编译区被烘死

> **状态**：已结案（2026-09-09）。机制已提炼进技能 `graph-capture-timing`，排查纪律进 `npu-remote-diagnose`。
> **来源**：2026-09 的 Claude Code 会话记忆（`~/.claude/projects/…/memory/`），2026-09-24 迁入本仓。
> 原文自带的「读法」提示保持原样：本文时间倒序，只有「2026-09-09 结案」一节是现状。

> ⚠️ **读法**：本文按时间倒序堆叠，只有紧接着的「2026-09-09 结案」一节是现状，其余都是
> 排查过程的历史记录（含多条已作废的假设，GDN 那条也在其内）。**不要把它当未结案的问题
> 引用**——2026-09-21 就因为只看了索引摘要、把它说成「没结案、分叉点在 GDN」被用户纠正。
> 文中「未解决：graph 下 MTP 接受率」也已结案，见 [`mtp-accept-graph.md`](./mtp-accept-graph.md)。

2026-09-04 起，`/mnt/share/weight/qwen3.5-397b-w4a4_multi` 单机 TP8+EP。

**权重是好的，别再怀疑它。** `GRAPH=0 MTP=3 QFA=1` 下 GPQA **88.89**，同时证明 QFA、
C8-MXFP8 cache、MTP 和新格式 V scale 加载全部正确。参见 [`2.4t-weight-history.md`](./2.4t-weight-history.md)。

**现象：prompt 超过 400 token 就只吐一个 EOS**（`completion:1`，不是截断）。三个条件缺一不可：

| EP | GRAPH | ≤400 | >400 |
|---|---|---|---|
| 1 | 1 | ✓ | ✗ |
| 1 | 0 | ✓ | ✓ |
| 0 | 1 | ✓ | ✓ |

## 2026-09-09 结案：空输出已修，MTP 接受率是另一条线

**修复落在 `junlin-c8-mxfp` @ `d33ddad19`（db5cb989d 之上单个提交，只动
`vllm_ascend/ops/fused_moe/fused_moe.py`）。** 长 prompt 输出已恢复正常。

`_fused_output_is_reduced` 喂给**三个**消费点，必须一起改：
`_maybe_reduce_routed_output_before_transform` / `_reduce_shared_output_if_needed`
/ `_maybe_reduce_final_output`。上游契约是成对的（要么 combine 规约了 routed、
shared 单独补，要么都没规约、最后对和补一次），**只改最终那处会让 shared 被规约
两回，回答从吐一个 EOS 变成整段乱码**——已实测踩中。三处共用
`_reduced_after_pre_transform` 判据，160 组全组合验证与「值不过期时的原逻辑」一致。

### 三个踩过的坑（别再犯）

1. ~~给 property 加 `torch._dynamo.disable`~~：vLLM 以 `fullgraph=True` 编译
   （`vllm/compilation/wrapper.py:150`），那里 graph break 是**报错**不是降级。
   只能用 custom op（dynamo 黑盒，不算 graph break）。
2. **op 必须写进单独的 `out`，不能就地改输入**：没有 shared expert 时调用方的
   `result` 就是 `fused_output` 本身，就地写会改坏它还持有的张量 → 整段乱码。
3. 探针用 `logger.warning` 并在模块导入时打一行版本标记；info 级探针在日志里
   一次都没出现过，白跑两轮。另外 editable 安装改了代码**必须重启服务**。

### 未解决：graph 下 MTP 接受率（下个会话）

| 配置 | 接受率 |
|---|---|
| eager | 60-70% |
| `CUDAGRAPH_MODE=PIECEWISE` | 正常 |
| `FULL_DECODE_ONLY` | **~24%** |

PIECEWISE 把 attention 和 GDN 一起放到图外，所以只知道是这两者之一进 FULL 图导致，
**还没区分开**。⚠️ 试过用 `GRAPH_KEEP_IN` 改 `CompilationConfig._attention_ops`
（绕开 `splitting_ops_contain_attention()` 断言必须包含全部 op 的限制）让其中一个
单独回图里，**结果两个方向都乱码且接受率更低，说明那个 hack 自身有副作用，实验
结论不可用**。诊断代码保存在 `origin/archive-c8mxfp-moe-reduce-wip`（tip
`e3c52485f`），要捡回去改。

观测到的 per-position 接受率是 `0.7 / 0.115 / 0.00`（junlin-qfa 上 2026-09-03 记的
`0.76 / 0.12 / 0.00` 是同一水平）。**此处不预设结论**：两条路都在 FULL 图内，
`vllm::unified_attention_with_output`（QFA）与 `vllm::qwen_gdn_attention_core`（GDN）
都是 `CompilationConfig._attention_ops` 的成员，PIECEWISE 会把两者一起切到图外。

⚠️ 已作废的归因：~~接受率低是 MTP 层没量化、v_cache_scale 走 127 兜底~~——eager 下
同一份权重能到 60-70%，权重侧解释不了 eager 与 graph 的差距。
⚠️ 已作废：~~`acl_graph.py` 的 draft 同步豁免~~——去掉 `not is_draft_eagle` 后接受率
没变，那道 barrier 不是原因，该改动已从历史中清掉。

## 2026-09-08 根因坐实：fused_output_is_reduced 在编译期被烘死

> ✅ **2026-09-08 已修复并实测通过**：`junlin-c8-mxfp` @ `8c3f2e233`，
> `EP=1 GRAPH=1 MTP=3 QFA=1` 下长 prompt 输出恢复正常。三个消费点改了两个
> （final + shared），第三个对本模型是死路、只加了 warning_once。
> 遗留：MTP 接受率仍是 `0.7 / 0.115 / 0.00`，那是独立问题（MTP 层在 checkpoint
> 里没量化，见 [`qfa-integration.md`](./qfa-integration.md)），与本 bug 无关。


决定性日志（`junlin-c8-mxfp`，`[moe-final-reduce]` 探针）：

```
shape=(16, 4096)    comm=MC2       passed_in=True recomputed=True  stale=False   # decode
shape=(16384, 4096) comm=ALLGATHER passed_in=True recomputed=False stale=True    # prefill
```

`MoERunner.forward`（`moe_runner.py:431`）在 forward 开头求值
`self._fused_output_is_reduced`，当成普通 Python bool 一路传到三个消费点。
vllm-ascend 把这个 property 覆盖成读 `_EXTRA_CTX.moe_comm_type`（`fused_moe.py:118`），
**而上游那次求值在编译区内，被烘死成捕获期 dummy run 的 MC2 值 True**；
超容量的 prefill 实际走 ALLGATHER，正确值是 False，于是最终 TP all_reduce 被跳过，
MoE 输出留着分片没规约。

⚠️ **本机 torch.compile 最小复现测不出这个折叠**（普通属性、custom op、
复刻 `__getattr__` 代理三种写法 dynamo 都重新求值了）。真机才暴露，别再拿本机
的「没折叠」当结论。

### 三个消费点必须成对修

`fused_output_is_reduced` 喂给：

1. `_maybe_reduce_final_output` → 决定要不要对 shared+fused 的和做 all_reduce
2. `_reduce_shared_output_if_needed` → 决定要不要单独规约 shared
3. `_maybe_reduce_routed_output_before_transform` 的 DP-only 分支（本模型走
   TENSOR_PARALLEL，碰不到）

上游契约是成对的：要么 combine kernel 规约了 routed、shared 单独补，要么都没规约、
最后对和补一次。**只修 1 会让 shared 被规约两次**（1 按重算值补了和，2 按过期的
True 又单独补了 shared），回答从吐一个 EOS 变成整段乱码。已实测踩中。

### 修复要点

决策包进 custom op（dynamo 黑盒，每次 eager 调用真实求值；被捕获的 decode 重放
捕获时的 MC2 决策，而 uniform decode 恒在容量内，是对的）。

⚠️ **op 必须写进单独的 out，不能就地改输入**：没有 shared expert 时调用方的
`result` 就是 `fused_output` 本身，就地写会改坏它还持有的张量，同样是整段乱码。
原始代码是把局部名重新绑到新张量，要保持这个契约。

⚠️ 探针用 `logger.warning` 并在模块导入时打一行版本标记：之前几轮 info 级探针
在日志里一次都没出现，白跑了两轮无法判断 op 到底有没有被调用。

## 2026-09-08 排除清单：最小复现已收敛到 GRAPH 一个开关

本轮逐个单变量测过，**全部仍然复现**，都不是变量：

| 试过的 | 结果 |
|---|---|
| MAX_NUM_SEQS 4 vs 100 | 都坏（容量 16 vs 400，阈值不跟着动） |
| TP 4 vs 8 | 都坏 |
| prefix caching 开/关 | 都坏 |
| `--no-async-scheduling` | 坏 |
| `TASK_QUEUE_ENABLE=0` | 坏 |
| **MTP 0 vs 3** | 都坏 |
| **CUDAGRAPH_MODE=PIECEWISE** | 坏 |

`ASCEND_LAUNCH_BLOCKING=1` 与图捕获冲突，起不来，试不了。

**最小复现：`GRAPH=1 QFA=1 MTP=0` + 长 prompt。** 没有投机解码、没有异步调度也照样坏，
后续实验一律用这个，变量最少。

**只有两件事能让它消失：`GRAPH=0`，以及在 GRAPH=1 下开 msprobe 插桩。**

### 本轮作废的两个假设

1. ~~async scheduler 覆盖 pinned buffer 的竞态~~。`CpuGpuBuffer.copy_to_gpu()` 确实是
   `non_blocking=True` 且源是持久 pinned buffer（`vllm/v1/utils.py:139`），
   `attention_v1.py:2628` 的 docstring 也自认过这个 race，但关掉 async scheduling 和
   算子下发队列都无效。
2. ~~draft-eagle 同步豁免~~。`acl_graph.py:262` 的
   `need_sync = FULL and not (is_draft_model and use_eagle)` 会在 MTP 下跳过
   `current_stream().synchronize()`，豁免理由指向的 `merge-eagle-graph` 在整个仓里
   只存在于那行注释本身、无任何实现。看着很像，但 **MTP=0 照样坏**，这条不成立。

### 2026-09-08 最终锁定：三条件合取，坏的是 eager prefill

**必须同时成立才坏**：① EP=1 ② 有捕获 ③ prompt 超过 `mc2_tokens_capacity`
（prefill 从 MC2 掉到 ALLGATHER）。缺任何一个都正常。所以「GRAPH 和 EP 不能共存」
的说法是错的，短 prompt 下两者共存正常。

`PREFILL_MC2=1` 只把容量从 16 抬到 2048（TP=4，上限 `_MC2_TOKENS_PER_RANK_LIMIT=512`
× tp），**边界推远而已**：prompt=2415 照样只吐一个 token。所以：

- ~~MC2 的 maxBs 太小~~ 已排除：maxBs 每卡 512 时照样坏。
- ~~HCCL window 不够~~ 已排除：`HCCL_BUFFSIZE=2560` 单独加无效。
- 开 `PREFILL_MC2=1` 必须配 `HCCL_BUFFSIZE=2560 GPU_MEM_UTIL=0.9`，
  否则 maxBs 4→512 让 `MoeDistributeDispatchV3` tiling 要不到窗口，EZ1008。

**坏的是 prefill 的 logits，不是 decode**：`completion:1` 说明第一个生成的 token
就是 EOS，而它来自 prefill 最后一个位置。**而 prefill 从来不在图里**
（FULL_DECODE_ONLY 只捕 uniform decode；PIECEWISE 下 900 token 不在 capture sizes）。
→ **是捕获阶段的 MC2 dummy run 留下的持久污染，伤到了之后 eager 的 ALLGATHER prefill。**

坏那次的第一个 token 分布是高熵不是崩坏：`<|im_end|>` 47.8%、top5 合计 61.8%、
长尾 38.2%。配合 **35B-A3B 同配置同条件不复现**（40 层 vs 60 层，hidden 2048 vs 4096，
moe_inter 512 vs 1024，两边 top_k 8/10 在 TP4 下都走 ALLGATHER、capacity 同为 16），
判断这是**程度问题不是二元错误**：每层扰动被 60 层累积后推翻了一个本就临界的分布。

### 下一步：之前的二分对照不干净，现在有干净的了

memory 上半部分那次逐层指纹是 `GRAPH=1` vs `GRAPH=0`，而 GRAPH=0 连编译一起关了，
所以那 1e-4 分不清来自编译还是捕获。**干净的一对是**：

- 坏：`CUDAGRAPH_MODE=FULL_DECODE_ONLY COMPILE_MODE=3 GRAPH=1`
- 好：`CUDAGRAPH_MODE=NONE COMPILE_MODE=3 GRAPH=1`

编译完全相同，只差捕不捕获。指纹要**顺序敏感**（旧的 `abs().sum()` 对 token 置换免疫）：
`(x.float().abs().sum(-1) * torch.arange(T, device=x.device)).sum()`，且要打在
**prefill 那一步**。

静态查过、没找到共享状态的地方：`_MoECommMethods` 是全局单例字典，MC2 与 ALLGATHER
两个 `MoECommMethod` 实例共存但各有自己的 token_dispatcher 和 prepare_finalize；
`PrepareAndFinalize*` 上有 `self.num_tokens` / `self.replace_allreduce` 这类
prepare 写 finalize 读的跨调用状态，但两条路是不同对象，互不覆盖。

### 2026-09-08 定论：捕获是根因，编译已洗清

| 配置 | 编译 | 捕获 | attention/GDN 在图里 | 结果 |
|---|---|---|---|---|
| FULL_DECODE_ONLY | 是 | 全图 | 是 | 坏 |
| PIECEWISE | 是 | 分段 | 否 | 坏 |
| **CUDAGRAPH_MODE=NONE + COMPILE_MODE=3** | **是** | **无** | 否 | **好** |
| GRAPH=0 | 否 | 无 | 否 | 好 |

**编译开着、只要不捕获就正常** → inductor / dynamo / 三个 fusion pass 全部无罪，
`COMPILE_BACKEND=eager` 和 `FUSION=0` 不必再跑。

**可用的规避方案（保住编译，只丢捕获）**：
`CUDAGRAPH_MODE=NONE COMPILE_MODE=3 GRAPH=1`。

**2026-09-08 复验：`EP=0 CUDAGRAPH_MODE=FULL_DECODE_ONLY MTP=0 TP=4 MAX_NUM_SEQS=4` 正常。**
EP=0 是那次唯一的新变量（TASK_QUEUE_ENABLE=0 / COMPILE_MODE=3 / TP4 / MAX_NUM_SEQS=4 /
MTP=0 都已各自在 EP=1 下证过救不了）。所以旧矩阵里「EP=0 好」在新分支新 CANN 上仍成立，
**EP 与捕获两个条件缺一不可**。

PIECEWISE 也坏说明被捕获的**非 attention 部分**才是问题所在（那两类 op 是
`_attention_ops` 里的切分点，piecewise 下留在图外）。结合 EP=0 好这条老结论，
最大嫌疑是 **MoE 的 EP 通信算子进图**（ALLGATHER 路径）。

### PIECEWISE 也坏 → QFA 图接入这条线被证伪

PIECEWISE 下 `vllm::qwen_gdn_attention_core` 与 attention op 都是切分点、留在图外，
`_get_qfa_metadata` 的 AICPU 调用也退回普通 eager（代码注释自陈）。**它照样坏，
说明根因不是 QFA 进图，也不是 AICPU metadata 在图里读到 stale 长度。**

因此以下已作废：~~AICPU metadata plan 在图里冻结~~、~~max_seqlen_kv=-1 的 tiling 被烘死~~。
QFA 接入确实缺 `graph_task_group_begin/end` + `graph_task_update` 那套（FIA 在
`attention_v1.py:1163` / `:645` 有，QFA 在 603340d01 有过、被 2e30035d5 删光），
**该补，但它不是本 bug 的根因**。

FULL 与 PIECEWISE 的公共集合只剩：dynamo 追踪 + inductor 编译。
GRAPH=0 与两者的差别也正是这个。所以嫌疑已收敛到**编译**，不是捕获。

### 下一步唯一没拆的：GRAPH 内部的编译与捕获

`GRAPH=0` 同时关了两样东西，脚本注释里写明了：cudagraph_mode 变 NONE，vLLM 跟着把
torch.compile 也关掉。`COMPILE_MODE`（19796ed 加的）就是为拆开它们而存在：
`GRAPH=1 CUDAGRAPH_MODE=NONE COMPILE_MODE=3` 是编译但不捕获。跑完要在日志里确认
mode 真是 3，别被 "Inductor compilation was disabled by user settings" 覆盖掉。

### msprobe 的两个坑（本轮查清）

- `model_runner_v1.py:365` 按 cudagraph_mode 选 dumper：NONE 用 `PrecisionDebugger`，
  否则用 `AclGraphDumper`。**eager 与 graph 两棵 dump 树出自不同的类。**
- `_dummy_run` 里 `_finalize_dump_data(dump=False)`（:3697）不写数据但推进 step 计数，
  profile_run 和每个 capture warmup 各吃一个 step 号，**step 索引跨模式不可比**，
  只能按 step 内的 token 宽度对齐。
- `acl_graph.py:252` 的 `logger.info_once("Replaying aclgraph")` 是 info 级，
  可用来确认 msprobe 有没有把图变相关掉。

### 脚本坑：exec 段里不能手工注释

`\` 续行后接注释行会就地结束命令（`#` 的注释吃到该行真实换行，注释内的尾部 `\`
不续行），后面那行变成独立命令；又因为是 `exec`，它永远不执行。把
`--additional-config` 注释掉会连带丢掉 `"${qfa_args[@]}"`，也就是
`--no-enable-prefix-caching`。已因此白采过一轮 msprobe 数据。

## 2026-09-07 实测推翻了此前的整条推理链

`[moe-sel]` 插桩打出决定性的一行：

```
[moe-sel] tokens=423 capacity=400 method=MoECommType.ALLGATHER soc=AscendDeviceType.A5 ep=True ep_world=8 over_capacity=True
```

**两处根本性的错**，此前所有分析都建立在它们之上：

1. ~~芯片是 A3~~ → **是 A5**。此前记的「由 MoeDistributeDispatchA3Tiling 报错确认是 A3」
   靠不住，于是一直照着 `_select_a3_moe_comm_method` 推，而真正生效的是
   `_select_a5_moe_comm_method`，两者逻辑不同。
2. ~~超容量切 ALLTOALL~~ → **切的是 ALLGATHER**。A5 分支是：
   ```python
   if num_tokens <= mc2_tokens_capacity and world_size > 1:  return MC2
   if world_size <= num_experts_per_tok:                     return ALLGATHER  # 8 <= 10 命中
   return ALLTOALL                                                             # 到不了
   ```
   `world_size=8 <= top_k=10`，**EP8 下永远走不到 ALLTOALL**。挂在 All2All 路径上的
   插桩注定全为 0，那不是异常是必然。

所以此前围绕 ALLTOALL 的排查（`pad_size` 笔误、`non_blocking` D2H、eager/图矛盾）
**全部打在空处**——那条路一次都没执行过。`pad_size` 那个上游 bug 是真的，但与本现象无关。

## 2026-09-07 二分结果：MoE 洗清，分叉点在 GDN

逐层指纹（`[moe-fp]` 打在 routed_experts.forward_impl 的公共入口，in/out 成对）：

| | GRAPH=1（坏） | GRAPH=0（好） |
|---|---|---|
| moe layer0 in | 6.094238e+05 | 6.094238e+05 | 逐位相同 |
| moe layer0 out | 5.944239e+01 | 5.944239e+01 | 逐位相同 |
| moe layer1 in | 5.285200e+05 | 5.285709e+05 | **分叉**（相对差 1e-4） |

**MoE 无罪**：输入输出两边都逐位相同。`[moe-ag]` 的路由掩码八个 rank 也逐位相同，
expert_map 假设作废。4 卡（EP4）同样复现，与专家切分无关。

由日志顺序可知 full attention 第一次出现在 moe layer2 之后（符合 interval=4，
模型 layer3 才是 full attention），**layer 0/1/2 都是 GDN**。而分叉恰好落在
moe layer0 out 与 moe layer1 in 之间，那段里只有 GDN——**分叉点就是 GDN**。

## 当前假设：GDN 的 fused chunk 探测在捕获期间失败

`vllm_ascend/ops/gdn.py` 的 prefill 有两条数值路径：

```python
use_fused_chunk = AscendGatedDeltaNetAttention._probe_fused_chunk() and get_pcp_group().world_size == 1
if use_fused_chunk:  ... _chunk_gated_delta_rule_fused(...)   # torch_npu 融合算子
else:                ... chunk_gated_delta_rule(...)          # Triton 实现
```

`_probe_fused_chunk` 是**类级缓存、每进程只探测一次**的 smoke call，里面有
`torch.npu.synchronize()`——**图捕获期间 stream 同步是非法的**（EE1016
"stream is captured"，插桩自己撞过同一个错）。失败被 `except Exception` 吞掉并把
`_fused_chunk_available = False` 永久钉死，于是 GRAPH=1 全程走 Triton 路径、
GRAPH=0 走融合算子，两条路径数值实现不同，1e-4 的差异逐层累积 60 层后输出崩溃。

**首次探测落在哪里，已静态查清（2026-09-07）：落在捕获阶段那一次 eager warmup 里，
GRAPH=0 下则落在第一次真实 prefill。** 两条链路缺一不可：

1. **profile run 触发不了它**。`_dummy_run` 建不建 attn_metadata 由
   `_should_build_dummy_attn_metadata` 决定，判据是 `force_attention or mode == FULL`
   （`model_runner_v1.py:3396`），profile 时两者都假。attention 层拿到 `None` 直接
   `output.fill_(0)` 返回，GDN 的真实路径压根没跑。
2. **捕获的 warmup 会触发它**。`_warmup_and_capture` 每个尺寸先跑一次
   `cudagraph_runtime_mode=NONE, force_attention=True` 的 eager warmup，
   `for_cudagraph_capture=is_graph_capturing` 是 False（`model_runner_v1.py:3576`），
   于是走 GDN builder 的普通 `build()`。那条路 `spec_sequence_masks is None`，
   `split_decodes_and_prefills(decode_threshold=1)`（`gdn_attn_builder.py:576`）
   把 MTP 的 query_len=4 判成 **prefill**，`num_prefills > 0` 成立，探测就在这里跑。
   capture 那一轮走 `build_for_cudagraph_capture`，是 spec 分支，算 decode，不触发。

**推论：日志里该看的是 `fused=`，不是 `capturing=`。** warmup 是 eager 的，
`cudagraph_runtime_mode=NONE`、没有 stream 捕获在进行，所以 GRAPH=1 那一次也会打
`capturing=False`，`torch.npu.synchronize()` 在那里是合法的——原文写的 EE1016 机制
大概率不成立。假设要靠两次启动的 `fused=` 值是否不同来证伪，而不是靠 capturing。
诊断在 `junlin-qfa` @ `12c0dee05`（`[gdn-probe]` / `[gdn-path]`）。

⚠️ **GDN 的 forward 走 `torch.compile(fullgraph=True)`**，插桩里任何 D2H（`.cpu()`、
`.item()`、`bool()`）都会让编译直接失败（`Data-dependent assertion failed`），
单测 `tests/ut/ops/test_gdn_layerwise_kv.py` 会拦下。诊断要放进
`_forward_core`／classmethod 这类 custom op 内部（`qwen_gdn_attention_core` 是
注册的 custom op，dynamo 黑盒、eager 执行）。

## 已排除的其它「捕获副作用」（2026-09-07 静态查清，别再重查）

捕获相对 GRAPH=0 只多跑了「每尺寸 1 次 eager warmup + 1 次 capture」
（`platform.py:1107` 把 `cudagraph_num_of_warmups` 设成 1）。它留下的持久状态里：

- **KV cache 没被写脏**。warmup 轮在 forward 之前就把整个 `slot_mapping.gpu` 刷成 -1
  （`model_runner_v1.py:3586-3589`），capture 轮跳过那句但沿用上一轮的全 -1。
  `reshape_and_cache` 照常被调用，slot 全是 `PAD_SLOT_ID` 被算子跳过。
  ⚠️ 这条**依赖 warmup 存在**：若把 `cudagraph_num_of_warmups` 改成 0，capture 会用
  `CpuGpuBuffer` 的零初值即 slot 0，真把 dummy KV 写进 block 0。
- **GDN 的 conv_state / ssm_state 写的是 block 0**，而 0 是 `NULL_BLOCK_ID`，
  `BlockPool` 启动时就摘掉了，永不分配给真实请求。何况全新请求 `context_len=0`
  → `has_initial_state=False`，conv 侧算子直接跳过预取，ssm 侧被 `clear_ssm_states`
  真存零（Triton kernel 是 `tl.store` 零，不是乘掩码）。
- **`_graph_params` / `_ATTN_KEYS_BUFFER` / FIA workspace 都读不到**：eager 前向的
  `cudagraph_runtime_mode` 是 NONE，进不了 replay 分支。
- **rope 的 `_cos/_sin` 全局缓存**捕获时被写成 position 0，但真实步每次 forward 都重写。
- **token dispatcher 无跨调用张量状态**：AllGather / MC2 两个类都已重构成用返回值里的
  `combine_metadata` 传递，`self` 上只剩只读配置（MC2 有个 `self.moe_expert_num`
  是 dispatch 写 combine 读，但同层内不会被打断）。两者不共享任何缓冲区。
- ⚠️ 唯一没排除的旁支：`_dummy_run` 里 `eplb_updator.forward_before()` **忽略了
  `skip_eplb=True`**（`model_runner_v1.py:3455`），开了 `dynamic_eplb` 的话捕获会推进
  换专家状态机。本次没开，但换配置时要记得。

## 证据本身的盲点：abssum 对 token 顺序免疫

`_dbg_moe_layer_fingerprint` 打的是 `abs().sum()`，对 token 轴的任意置换、对逐元素
符号翻转都不变。所以「layer0 in/out 逐位相同」只证明输出的**集合**没错，不证明
**顺序**没错；而紧跟其后的 GDN 是沿 token 轴的递归扫描，是全模型对顺序最敏感的一段。
要坐实「MoE 无罪」得补一个顺序敏感的量，例如按位置加权：
`(x.float().abs().sum(-1) * torch.arange(T, device=x.device)).sum()`。

## 已作废的嫌疑：ALLGATHER + EP 的路由掩码

`token_dispatcher.py` 的 `TokenDispatcherWithAllGather.token_dispatch`，EP 开关的分叉点：

```python
if expert_map is not None:            # EP=1
    mask = expert_map[topk_ids] != -1
    topk_weights = topk_weights * mask        # 不属于本 rank 的专家权重置零
    first_expert_idx = get_ep_group().rank_in_group * self.num_experts_local
else:                                 # EP=0
    first_expert_idx = 0
    global_num_experts = self.num_experts_local
```

若这里权重被整体清零 → MoE 输出全零 → 只吐得出 EOS。三个"好"条件都对得上：
EP=0 不进这个分支；≤400 走 MC2 不碰这段。**GRAPH=0 为什么好仍未解释**，是下一个要答的。

已加 `[moe-ag]` 插桩（`junlin-qfa` @ `350d65e8e`）打掩码前后的 `sum_before/sum_after`、
`nonzero_after`、`ids_range`，判定权重是否被清光、topk_ids 是否越出 expert_map 下标。

## 容量的两个用途（别只记住第一个）

`mc2_tokens_capacity` 除了在 `select_moe_comm_method` 里决定通信方式，还在
`TokenDispatcherWithMC2.__init__` 里定 MC2 算子的 maxBs：
`num_tokens_per_tp_rank = capacity // tp_size`（400/8=50），`_max_global_bs = 那个 * ep_world`。
实测 `[moe-mc2] ... per_rank=50 global_bs=0`——global_bs=0 是 uniform 模式（传 mc2_mask）。

容量算法：`enable_prefill_mc2` → `max_num_batched_tokens`；否则有 capture sizes 时取
`max_cudagraph_capture_size`，没有时取 `max_num_reqs × uniform_decode_query_len`。
**GRAPH=1 与 GRAPH=0 两式同值都等于 400**（100×4 = 最大捕获尺寸），因为 `cudagraph_mode=NONE`
时 vLLM 的 `_set_cudagraph_sizes()` 整块被跳过、capture sizes 留空。推论：调
MAX_NUM_SEQS / MTP 会让两边容量一起变，**不能靠它分离变量**。
`_MC2_TOKENS_PER_RANK_LIMIT=512` 是硬常量，tp=8 下 4096 是天花板。

## 插桩现状（junlin-qfa @ 350d65e8e）

`[moe-sel]` 无论选谁都打（**这条是地基**：其余插桩都挂在具体某条通信路径上，
压根没选中时它们一起为 0，与"插桩失效"无法区分，已因此白读过一轮日志）；
`[moe-mc2]` 打 MC2 的 capacity/per_rank/global_bs/fits；`[moe-ag]` 打 AllGather 的路由掩码；
`[moe-prep]/[moe-disp]/[moe-fp]` 挂在 All2All 上，A5+EP8 下永远不触发，可以摘。
`[moe-disp]` 另带 `stale=` 列，判定 `_preprocess` 里两处
`.to(cpu, non_blocking=True).numpy()` 是否在拷贝落地前被读走（那段唯一的
`torch.npu.synchronize()` 挂在 `num_local_experts <= 1` 分支上，每卡 64 专家走不到）。

⚠️ 开 `PREFILL_MC2=1` 后 MC2 maxBs 50→512，MoeDistributeDispatch tiling 要 4433MB 窗口，
profile_run 阶段 EZ1008。报错文案写「HCCL_BUFFSIZE_EP is too SMALL」**但抬那个变量没用**，
真正的杠杆是 `HCCL_BUFFSIZE=2560`（→5120MB），配 `GPU_MEM_UTIL=0.9`。

## 排查方法论上的三个坑

`C8=0` 在 `junlin-qfa` 上是哑开关（`VLLM_ASCEND_DISABLE_C8_MXFP` 只在 `junlin-qfa-c8switch` 定义）；
日志去重键不含真正想区分的维度就会骗人（`[qfa] capture` 按 `(state, serves)` 去重，
只打一行不代表只走过一种长度）；`scripts/local/run_cpu_ut.sh` 传路径参数时走 partial 模式
**不回传 pytest 退出码**，见 技能 `npu-remote-diagnose` 的「脚本交付」与 `scripts/local/README.md`。

**还欠着**：MTP 层 `v_cache_scale` 全 127 未加载，per-position 接受率 `0.76/0.12/0.00`，
只损失速度不损失正确性，见 [`qfa-integration.md`](./qfa-integration.md)。
