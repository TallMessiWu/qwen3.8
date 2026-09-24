# QFA 接入全过程（vendor → eager → 进图 → MXFP8 cache）

> **状态**：**部分过期**。本文主体写于 2026-09-02/03，讲的是 `junlin-qfa` 那条分支；
> 2026-09-08 起真机已经切到 `junlin-c8-mxfp` 系，两条分支的 attention 实现差别很大。
> **来源**：2026-09 的 Claude Code 会话记忆（`~/.claude/projects/…/memory/`），2026-09-24 迁入本仓。
> 原文自带的 2026-09-08 过期提示保持原样，下面是更完整的导读。

## 引用本文前先看这张表

分支血缘（右侧是现状，左侧是历史）：

```
junlin-qfa           基于 upstream/main，本文描述的就是这一条
  └─ junlin-c8-mxfp          跟随上游 PR 15484(C8 MXFP8 KV cache + QFA + MTP + PD 分离)
       ├─ junlin-c8-mxfp-16278   同一批补丁 rebase 到更新的上游
       └─ junlin-c8-mxfp-16614   从上游 PR 16614 头部派生；+2 个提交修 hybrid KV 容量
```

| 本文的哪部分 | 现在还能不能用 |
| --- | --- |
| vendor 六道门、binding、契约体检、进图纪律 | ✅ 已提炼进技能 `vendor-ascend-op`（**看技能，别看本文**） |
| descale shape 表、`max_seqlen_q` 语义、tilingKey 六维 | ✅ 已提炼进 [`qfa-op-contract.md`](../reference/qfa-op-contract.md) |
| 「值在错误时刻被固定」两条真机故障 | ✅ 已提炼进技能 `graph-capture-timing` |
| 本文对 `junlin-qfa` 分支结构与提交链的描述 | ⚠️ 该分支的历史**被重写过至少两轮**，所有更早的提交号都已失效；引用前先 `git -C vllm-ascend/<worktree> log --oneline` 核对 |
| 本文关于 QFA 双算子来源的说法 | ⚠️ 只对 `junlin-qfa` 成立：它从 csrc 编出算子。`junlin-c8-mxfp` 系从外部 `cann_ops_transformer` 包取，csrc 里没有算子源码 |
| 本文里 `archive-*` 相关的一切 | ⛔ **被推翻的设计，只作代码考古**：四/五平面 cache、`attention_qfa.py` 独立 backend 子类、M0→M1→M2→M3 里程碑流水线。看了会被带偏 |
| 本文提到的 `junlin-qfa-c8switch` / `feat-qfa-dump` / `debug-moe-comm-tokens` | ⛔ 2026-09-10 连同全部 `archive-*` 备份一起从本地和 fork 删除；配套的 `scripts/bench/replay_qfa_dump.py` 同日删除 |

### 文中脚本路径对照（2026-09-24 核对，正文不再逐处改）

本文按当时的样子保留，下面是那些路径现在的下落：

| 文中写的 | 现在 |
| --- | --- |
| `scripts/bench_qfa_vs_fia.py` | 改名为 [`scripts/bench/test_qfa_vs_fia.py`](../../scripts/bench/test_qfa_vs_fia.py) |
| `scripts/debug/check_c8_mxfp_weight_support.py` | 归位到长期复用目录：[`scripts/checks/c8_mxfp_weight_support.py`](../../scripts/checks/c8_mxfp_weight_support.py) |
| `scripts/debug/test_qfa_graph_capture_npu.py`、`check_smoke_gate_offline.py`、`test_qfa_fullgraph_repro_npu.py`、`scripts/bench/replay_qfa_dump.py` | **已删**（前两个在主仓 `4c1da3d`，`replay_qfa_dump.py` 随 `feat-qfa-dump` 一起）。本文说「图捕获的验证角色由 `test_qfa_fullgraph_repro_npu.py` 承接」——那个脚本后来也删了，要重做得把捕获侧与回放侧一起写回来 |
| `scripts/debug/verify_expert_split_axis.py`、`check_quant_desc_qwen35_moe_text.py` | **已删**，见 [`2.4t-weight-history.md`](./2.4t-weight-history.md) 的重写方法 |
| `scripts/debug/diag_qfa_metadata_size.py` | 已删；同类能力现在是 `scripts/checks/qfa_metadata_capacity.py` |

判断依据是 `AGENTS.md` 约束 3：只为回答当下这一个问题的进 `scripts/debug/`，**结案即删**——
所以本文提到的 `scripts/debug/*` 绝大多数已经不在，别再照着找。

## 读源码得到的事实（摘自同期 plan 文件，2026-09-24 收录）

下面几条读自 `feat/qfa-mxfp8-attn` 时代的源码，**函数名与行号都已失效**；但机制层面的
判断能用来识别同类陷阱，所以在 plan 文件本身不搬之后单独留档。

前两条已经写进 [`qfa-op-contract.md`](../reference/qfa-op-contract.md)，这里只留结论：
metadata plan 与层无关（一步算一次、每层共用是净收益，也是进图的前提）；
QFA 主算子的 host tiling 与序列长度无关（动态切分全在 metadata 这个设备缓冲里）。

**① 捕获期 `attn_state` 是 `ChunkedPrefill`，不是 decode。**
`model_runner_v1.py` 里 dummy/capture 走 `DecodeOnly`，但对 **MTP 且非 MLA** 的模型
（Qwen3.8 正是）改成 `ChunkedPrefill`。BF16 FIA 不在乎这个差别，QFA 在乎：
读侧 `layout_q_descale` 会从 `N2TGD` 翻成 `TND`，写侧会走带 Python 逐请求循环的
bulk 写路径 ⇒ **捕获下来的图形状完全不对，重放每一步都错**。分支选择必须与
`attn_state` 解耦。

**② `spec_verify` 挂在 `attn_state` 上 ⇒ 捕获期会静默丢掉一次写。**
它要求 `attn_state == SpecDecoding`，而捕获期是 `ChunkedPrefill` ⇒ 记录成单次写、
图模式下「两次写的 restore」静默消失，质量退化原地回来且不报错。必须改成静态判据
（目标模型 且 `num_speculative_tokens > 0`）。

**③ 惰性分配必须在捕获前强制初始化。**
当时 `_qfa_attach_fp8_cache`（对整个 cache 做 `zero_()`，10+ GiB）与 `_get_qfa_mask`
（2048×2048 int8）都是首次调用时才分配，落进捕获区就是灾难。这与
`graph-capture-timing` 里那条「捕获块内 `torch.zeros` 会把清零本身录成图节点」是同一族，
扫这类代码用 `scripts/checks/scan_capture_timing.py`。

**④ 一条关于脚本的可信度教训。** 当时那个图捕获实验脚本只有一个提交、无后续修正，
而本仓每个探针脚本上机后都需要 1~2 次修正提交 ⇒ 大概率从没真正跑过；而且它
**只捕获主算子**，没覆盖 `npu_dynamic_mx_quant` / `index_put_` / `searchsorted`，
也就是整条写路径在图里能不能活根本没验。**引用别人（或自己）留下的实验结论前，
先看它有没有被后续提交修正过。**

---

> ⚠️ **2026-09-08 更新**：服务器已切到 `junlin-c8-mxfp` 分支，并换了 CANN 包。
> 下文"服务器 editable 安装的就是 junlin-qfa"已不成立；`junlin-c8-mxfp` 的 attention
> 实现与 junlin-qfa 差异很大（QFA 双算子从 cann_ops_transformer 取，长度从持久缓冲区
> 在设备侧派生），本文的 QFA 细节不要直接套到那条分支上。换 CANN 后先跑
> `scripts/checks/qfa_op_contract.py` 确认算子签名没漂。

QFA（QuantFlashAttn）接入现状（2026-09-02 核对）。**在途分支只剩 `junlin-qfa` 一条**，
基于 upstream/main @ bb6d5b82e，worktree `vllm-ascend/junlin-qfa`（服务器 editable
安装的就是它）。图那条线已并进来，`junlin-qfa-graph` 这个分支名**已不存在**。

tip `25589d5c8`（2026-09-07 起），主链：`1622a77f1` vendor 双算子（binding 含 out
变体）→ `55720da62` 合入 Tame21/cache-c8 的 MXFP8 KV cache 框架 → `40f92f52e` causal
主路径 FIA→QFA、直读 MXFP8 cache → `fe58a86a4` QFA 进 FULL 图、走 task group 与 update
重发 → `0ae367070` 每层 quant_type 的 MXFP8 C8 权重适配 → `25589d5c8` MoE All2All 插桩。

**原 `a95e3d3bc`「集中调试日志」已于 2026-09-07 用 rebase 移出历史**（刷屏太狠），
不是 revert，历史里查不到。它完整保存在 `origin/archive-qfa-debug-logs`（tip
`552ebeaf4`）和本机 `~/qwen3.8-backups/0001-loud_sound-chore-qfa.patch`，
要回它：`git cherry-pick a95e3d3bc` 或 `git am` 那个 patch。
⚠️ 里面 `mxfp_c8.py` 那 9 行 `logger.info_once` 不刷屏，却是 MTP 层 v_cache_scale
全 127 未加载那笔账的唯一证据来源；只想止刷屏的话单独把它捡回来即可。
⚠️ `_trace_qfa_plan` 里 `metadata[:4].tolist()` 是 D2H 同步，曾意外把前几个 decode 步
串行化、掩盖过 AICPU 写 plan 与刷缓冲区的竞态。该竞态已由 `fe58a86a4` 的 plan_ready
事件修掉，所以摘掉安全——但若将来回退 `fe58a86a4` 却把这层日志捡回来，竞态会重现。

⚠️ 这条历史被重写过至少两轮，**所有更早记下的提交号都已失效**：`ce66a81a0` /
`9a7876e93` / `aa0b6ef3f`（旧 junlin-qfa，现为 `archive-junlin-qfa-eager-baseline` 的 tip）、
`2c5ffa432` / `c85c545ec`（旧 junlin-qfa-graph）、以及更早那 6 个
（24be94e05 / c16f0de10 / 1a21c241d / 208913440 / 5f2b644ca / 53f244640）。
**引用提交号前先 `git -C vllm-ascend/junlin-qfa log --oneline` 核对。**

- **`archive-qfa-graph-unsquashed`**（tip `1c8e3a70e`）、**`archive-qfa-graph-taskupdate`**
  （tip `b0d230f42`）、**`archive-junlin-qfa-eager-baseline`**（tip `aa0b6ef3f`）
  —— 重写前的完整历史存档，**不是被推翻的设计**，与下面那批 archive-* 不同类。
  ⚠️ unsquashed 那份里 `_qfa_quant` 还是旧名，且 `scatter_mxfp_v_cache` /
  `scatter_mxfp_v_scale_cache` 被误删了——都已在现分支修回，别拿存档当参考。

⛔ **所有 `archive-*` 分支（archive-qfa-m1-fiveplane / -m2 / -m3 / archive-qfa-mxfp8-attn）
都是被推翻的设计，只作代码考古，看了会被带偏。** 连带失效的还有：`attention_qfa.py`
独立 backend 子类、四/五平面 cache、M0→M1→M2→M3 里程碑流水线、
`scripts/tests/test_qfa_backend_*.py` 四件套。那四个脚本连同孤儿
`scripts/debug/check_smoke_gate_offline.py`、无法 bootstrap 的
`scripts/debug/test_qfa_graph_capture_npu.py`，已在主仓 `4c1da3d` 删除；
图捕获的验证角色由 `scripts/debug/test_qfa_fullgraph_repro_npu.py` 承接。

**已完成**
1. vendor：官方 ops-transformer master @ 14cf794f3 的 `quant_flash_attn` +
   `quant_flash_attn_metadata`（AICPU 伴生，必须成对、两次调用参数逐项一致）原样
   进 csrc/attention/，各自内嵌 attention/common 依赖头（**永远不要覆盖 csrc 共享
   common**，那是旧裁剪快照）。binding `torch.ops._C_ascend.npu_quant_flash_attn{,_metadata}`。
   单算子 8 case 含 golden 在 A5 实机全绿。
2. 替换 FIA（2026-08-30）：`attention_v1.py` 的 causal 主路径改调 QFA，27B 满配
   （MTP + aclgraph）实机可服务，多模态回答正常。
3. 开关（2026-08-30，commit c16f0de10）：恢复 `VLLM_ASCEND_ENABLE_QFA`（envs.py，默认 0），
   `AscendAttentionBackendImpl.__init__` 读一次存 `self.enable_qfa`（热路径不能每步
   os.getenv），QFA 主体抽成 `_forward_qfa`，FIA 调用按上游原样恢复。`scripts/27B.sh`
   的 `QFA=1` 分支这才真正生效。
4. 图脚手架（2026-08-30，现 `aa0b6ef3f`）：**代码就位但图里实际仍是 FIA**，见下方 🔴。
   metadata 提到 builder
   （`_attach_qfa_inputs`），捕获期自己持一份 buffer（`QFAGraphBuffers`），replay 前
   由 `_update_qfa_graph_buffers` 在 update_stream 刷内容再 record 事件放行。
   **不需要 FIA 的 workspace / `.out()` / `graph_task_update`。**
5. 对比脚本（主仓 `scripts/bench_qfa_vs_fia.py`）：两次测量各落一份 JSON，`--compare`
   出对照表；prompt 带唯一前缀使前缀缓存对两边都无效，27B.sh 另加 `NO_PREFIX_CACHE`。

**关键：现在只是探针，不省显存。** KV cache 仍 bf16，每层每步现场量化 K/V 再喂算子，
所以开着必然更慢。下一步才是让 cache 以 MXFP8 存储。

**实机已验配置**（27B.sh）：QFA=0 下 MTP3+GRAPH1（默认）/ MTP0+GRAPH0 / MTP3+GRAPH0 均可服务。
**QFA=1 需要 `GPU_MEM_UTIL=0.85`**：0.95 下 capture 报 no available memory（图池要单层
bf16 KV cache 的 1.28 倍）。0.85 时 GRAPH=0 与 GRAPH=1 都能正常服务、回答正常（2026-08-30）。
`ASCEND_LAUNCH_BLOCKING=1` 与 **aclgraph 捕获**互斥（不是与 QFA 互斥），要用它调试须配 GRAPH=0。

✅ **里程碑：QFA 进 FULL 图跑通（2026-09-01 实机）**。`MTP=0 GRAPH=1 QFA=1`，权重
Qwen3.5-35B-A3B-mxfp4-c8，服务正常、`scripts/curl.sh` 的多模态回答正常。
**这是子目标「decode 图捕获 QFA」第一次达成**，attention 终于留在图内。

怎么做到的（三步，都在 `junlin-qfa-graph`）：
1. `e3223340f` 给 QFA 加 `.out` overload——纯 binding 层，不碰 kernel（aclnn 本来就收
   `attnOut`/`softmaxLse`）。`quant_flash_attn_torch_adpt.h` 加薄封装、
   `torch_binding.cpp` 加 schema、`torch_binding_meta.cpp` 加 meta。
2. `963cf77de` 把 `full_graph_qfa` 从「捕获期自持 buffer + 每步 copy_」改成 FIA 那套：
   `.out()` 包进 `graph_task_group_begin/end` 拿 handle，`_update_qfa_graph_params`
   在每次 replay 前用本步的 metadata / block_table / 长度重发整个调用。
   `QFAGraphBuffers` 与四个 `copy_` 整套删除。`q_fp8`/`q_descale` 例外，原样回传——
   图内每步都会从本步 query 重算它们。输出直接写进 forward 传入的 `output` 切片。
3. `abebfe9c0` 加捕获凭据日志（见下）。

**验证捕获是否真的发生，看三行**（`_qfa_serves` 的 `serves=True` 证明不了，它在 eager
路径也打——`5f2b644ca` 那次乌龙就是这么骗过去的）：
- `[qfa] captured op #N into task group at tokens=S` —— **只有 `full_graph_qfa` 拿到
  handle 后才可能打**。实测 9 个尺寸 × 10 层 = 90 个 op。
- `[qfa] update tokens=S -> updating {'ops': 10, 'handles': 10, 'events': 10}` ——
  ops 必须等于全注意力层数。出现 `skip: nothing captured` 就是没捕上。
- `[qfa] plan #N ... header=(1, 0, 128, 256)` —— 捕获期与 replay 期必须相同。

✅ **MTP + FULL 也跑通了（2026-09-01，`9633c8ac2`）**：`MTP=3 GRAPH=1 QFA=1`，崩溃计数 0。
实测 target 32 个图尺寸 × 10 个 op（10 个全注意力层）、draft `ops=3`
（3 个 draft 步 × `mtp.layers.0` 那 1 层），两边 handles/events 都对得上。
⇒ **QFA + MXFP8 cache + MTP + FULL 图全开可服务**，attention 全程在图内。
改动就是把 draft 那三处硬编码拆掉：调用点的 `not (qfa_capture and is_draft_model)` 去掉、
`_qfa_graph_params()` 按 `is_draft_model`/`is_draft_model_prefill` 选注册表、
`_qfa_steps_per_op()` 把 `draft_attn_metadatas` 按 `(draft_step, key)` 展平对齐捕获顺序
（FIA draft 分支一直这么做）。`plan_ready` 的等待要挪进循环——draft 每步有自己的 plan。

⚠️ **`MTP=0` 下 target 的 decode 图以 `DecodeOnly` 捕获，不是 `ChunkedPrefill`**——
下面那条 🔑 只在 MTP 开着时成立（`SpecDecoding is only designed for mla`，MTP+非 MLA
才会退成 ChunkedPrefill）。判断「捕获状态对不对」时要带这个限定词。

**不必重新验证的硬事实**
- **metadata 的 `v_descale` 是「必须非空、但 PA_BBND 下没人读」**（两层，只看一层会栽）：
  ① aclnn 入口 `quant_flash_attn_metadata_check.h:217` 硬性要求非空，quantMode=1 传 null
  直接 `EZ9903 ... vDescale must be provided`；秩检查只覆盖 TND(4D)/PA_BNBD(5D)/PA_NZ(6D)，
  **PA_BBND 整个 if/else 链都不匹配，无任何形状约束**；② AICPU
  `quant_flash_attn_metadata_aicpu.cpp:230` 读 dim0 也只在 TND 下。
  ⇒ paged 状态传一个 5D 最小尺寸的 e8m0 占位即可，同一步所有全注意力层共用一份 plan，
  且能在 builder 里、任何层量化之前算出来——图路径成立全靠这条。
  算子自己分配 4096 int32 输出（`METADATA_NUM_INT32`，torch_adpt.h:30），Python 侧改不了大小。
- **捕获期与 replay 期的张量地址不保证一致**：MTP proposer 的 `dummy_run` 用
  `query_start_loc_group[draft_index]`，真实 `_propose` 用 `self.arange`。FIA 正是靠
  `graph_task_update` 重绑参数才不受影响。所以进图的算子要么重绑，要么读捕获期自己
  持有的 buffer（QFA 走后者）。
- **MTP proposer 先把所有 draft 步的 metadata 建完再逐步跑**
  （`llm_base_proposer.py` `_propose` 的 `multi_steps_attn_metadata` 循环）。⇒ 每步的
  输入张量必须各自分配，共用一份 builder buffer 会让所有 draft 步读到最后一步的值。
- **`max_seqlen_kv` 会经 `AdjustSinnerAndSouter` 影响切分计划**，且被烘进捕获的算子，
  metadata 与主算子必须同值 ⇒ 进图后只能用常量（现取 `max_model_len`）。
- **out-of-tree backend 的 `get_name()` 必须返回 `"CUSTOM"`**：vllm 0.27 的
  `Attention.__init__` 做 `AttentionBackendEnum[get_name()]`，闭合枚举，插件侧唯一槽位
  就是 CUSTOM。backend 身份靠 `platform.get_attn_backend_cls()` 区分。`AscendMLABackend`
  返回 `"ASCEND_MLA"` 不炸是因为 MLA 不走这条查表，别照抄。
- **V 的 scale 沿序列分组，K 的沿 D 分组**（决定 MXFP8 cache 怎么设计）：同一
  (head, dim) 上连续 32 个 token 共享一个 e8m0，v_descale dim0 由 kv 长度算出且 kernel
  会校验；k 没有这类校验。⇒ decode 逐 token 追加会改变未满组的最优 scale，V 的写入
  路径必须处理这个时间依赖（留 bf16 底稿 / 只留当前窗口底稿 / 完全不留并 clamp）。
- **`actual_seq_lengths_kv` 语义随 attn_state 变**：PrefillNoCache 下是累加值，
  其他状态是逐序列长度。B>1 时混用即错。TND 必须传 `cu_seqlens_kv`（不传直接被拦），
  PA_BBND 用 `seqused_kv`。
- **hybrid 模型下调度块 ≠ kernel 块**：Qwen3.5 上 `unify_kv_cache_specs` 把 attention
  调度块从 128 拉到 1536 对齐 GDN 页，kernel 侧恒 128；cache 布局 / block_table /
  slot_mapping 全按 kernel 块编址。
- **metadata 的 split-plan buffer 固定 4096 int32 恒够用（源码证明，不是估计）**：vendor 的
  AICPU 里写死 `param.l2Byte = 0`（`quant_flash_attn_metadata_aicpu.cpp:223`），而
  `CalcGridInfoSection` 一看见 l2Byte==0 就 `sectionNum = 1` 直接返回
  （`section_stream_k_impl.h:316`）⇒ **sectionNum 恒为 1，与 batch / heads / 序列长度全无关**。
  布局是 head 16 + sectionNum×(AIC 36 + AIV 72)×16 uint32（`quant_flash_attn_metadata.h`），
  即恒定用 1744/4096。官方 `((36+72)*batch*heads+1)*16` 就是同一公式取 sectionNum ≤ batch*heads
  的上界。⇒ 「plan 缓冲区被大 batch 写爆 → 越界 → invalid GM address」**不可能**成立；
  `GenMetaData` 确实没有边界检查（只有 release 会编掉的 assert），但永远走不到那里。
- **`full_graph_qfa` 只在 `_EXTRA_CTX.capturing` 下被调用**（`forward_fused_infer_attention`
  开头），所以捕获期禁止在它里面做 D2H；要打 plan 的日志得放在 builder 侧。
  **`attn_mask` 不是地址稳定性嫌疑**：`get_splitfuse_attn_mask()` 返回 singleton 缓存的
  2048² int8，捕获期与 replay 期是同一个对象。
- **`_qfa_quant_q` 只给 q 用，不是残留**：K/V 从 MXFP8 cache 里出来就是量化好的，
  没人再动它们；q 不在 cache 里，是图内 QKV 投影每步现算的 bf16，而 QFA 只收 MXFP8，
  所以 q 每步必须量化一次——他们的 FIA 路径同理，只是要的 scale 形状不同
  （FIA 用 `npu_dynamic_mx_quant` 的原始输出，QFA 要 TND 的 `(T, N, D//64, 2)`）。
- **FIA 的 MXFP8 只吃 D=64/128，已在真机坐实**（2026-08-31，不再是文档转述）：D=256 直接
  `Invalid_Argument_Tensor_Shape(EZ0010) ... In the MXFP8 full quant scenario, the axis D of
  query and value must both be 64 or both be 128`，是算子 tiling 的硬校验，不是 CANN 版本问题。
  ⇒ 同事的 `cache-c8` 分支（走 FIA v2）在 D=256 的模型上从来跑不通 attention；选 QFA 的理由成立。
- **QFA 直读 MXFP8 cache 在真机跑通（eager，2026-08-31）**：Qwen3.5-35B-A3B-mxfp4-c8
  （hybrid、D=256、kv_heads=2、block_size 512、1612 blocks），`GRAPH=0 QFA=1 MTP=0` 可服务，
  curl 正常。`serves=True` 覆盖 PrefillNoCache(max_q=1552) 与 DecodeOnly(max_q=1)。
  同一位置 FIA 挂、QFA 过——这是替换的第一份真机证据。
- **plan header 恒定，「捕获/replay tiling 不一致」这条假设家族可以划掉**：实测
  `header(sections,is_fd,m_base,s2_base) = (1, 0, 128, 256)`，在 max_q ∈ {1, 17, 25, 1552}
  下**完全不变**。sections=1 与源码推断一致（`param.l2Byte = 0` ⇒ sectionNum 恒 1）；
  m_base/s2_base 由 `AdjustSinnerAndSouter` 算出但被钉死的 max_seqlen_kv=133120 主导，
  与 max_q 无关。⇒ 进图若仍崩，不是 plan 与烘死 tiling 对不上。
  ⚠️ 仅在 batch=1（`cu=(2,)`）下测过，多请求未验。
- **V 的静态 scale：27B 没有；35B 有，已接上（`c4affa025` / `f685bb13d`）**。
  35B 的形态已查实：`v_proj.v_scale` `[512,1]` uint8、**10 个全注意力层齐全**、
  值分布集中在 117~122（另有 4~84 个 0），而 `v_proj.weight_scale` 是 `[512,64]`（per_block，
  Wv 自己的）、`k_proj.k_scale` **不存在**——完全符合 `K_DYNAMIC_V_STATIC` 配方。
  ⇒ **`v_proj.v_scale` 确是 cache 的 V scale，不是 Wv 的**（上游 vLLM 的
  `get_cache_scale_mapper` 也把这个名字归类为 KV cache scale）。
  之前读不进来的真正原因：**上游那条正则先把它改名成 `attn.v_scale`**
  （`WeightsMapper._map_name` 里 regex 在 suffix 之前跑），而我们注册的参数叫 `v_cache_scale`
  ⇒ 落空，一直是 127 兜底。所以映射要挂在**改名后**的 `.attn.v_scale` 上，挂 `.v_proj.v_scale` 无效。
  `v_scale == 0`（minmax 校准该通道 absmax 为 0）按中性 127 处理——`2^-127` 的倒数是 `2^127`，
  推理时该通道只要不是严格 0 就会 inf。
- **别高估这个 scale 的精度收益**：E4M3 是浮点，3 位尾数的相对精度与数值大小无关，
  per-channel 的 2 的幂 scale 主要买的是**防溢出/防下溢的余量**，不是尾数精度——这与 int8
  完全不同。所以 127 兜底下「回答看起来正常」是解释得通的，不是错觉。真正的风险在
  下溢通道（layer 39 有 84/512 个 0）。
- ~~**V 的静态 scale：27B 没有，35B 有但读不进来**~~（已解决，见上）：27B 的 v_proj 只有 `weight` / `weight_scale`；
  35B 有 `model.…v_proj.v_scale`，`[512, 1]` uint8（= kv_heads 2 × head_dim 256）。但代码里
  只映射 `.v_proj.kv_cache_scale`，全仓没人认识 `.v_proj.v_scale` ⇒ 两份权重现在都走
  `mxfp_c8.py` 的 127 兜底（scale=1.0，等于 V 只截断不缩放）。修它要两处：suffix_map 加名字，
  以及 `[512,1]` vs 参数 `(512,)` 差一维会撞 `_quant_weight_loader` 里的 size 断言。
  ⇒ **在此之前所有实验都不能谈精度**，只能谈通不通、崩不崩、省多少显存。
  27B 要拿到真 scale，得让 ModelSlim 按 35B 那份 `best_practice.yaml` 里的
  `type: dynamic_cache`（scope per_channel / dtype mxfp8）重量化一次。
- **进图崩溃的算子名终于读到了：`fault kernel_name=QuantFlashAttn`**（2026-08-31，plog 在
  `/root/ascend/log/run/plog/`，不是 debug/plog）。stdout 的 traceback 指向 builder 里的
  `torch.tensor(...)` H2D，是假的；真信息是 `EE9999 ... rtEventRecord execution failed,
  reason=the model stream execute failed` —— 图 replay 那条流执行失败，下一次同步才冒出来。
- **根因（已修，`253dd1d42`）：AICPU 写 plan 与刷进图缓冲区之间没有流依赖。**
  `_attach_qfa_inputs` 在 builder 流上跑 AICPU metadata 算子；`_update_qfa_graph_buffers`
  在 `update_stream` 上把结果 `copy_` 进捕获期 buffer，两条流无依赖 ⇒ 拷贝可能跑赢 AICPU 的写，
  捕获的算子读到半成品 plan。FIA 没暴露是因为它的输入是普通 H2D，窗口极窄；AICPU 延迟又长又飘。
  **不能用 `update_stream.wait_stream(current_stream)`——图已入队并停在事件上，会死锁**；
  正解是在 plan 入队后 `record_event()`，`update_stream.wait_event()` 只等那一点。
  与归档分支 `3108031eb`（507014）同一类，那次用的是 host-wide `torch.npu.synchronize()`。
  🔎 旁证：崩溃发生在第二三个 decode 步，而第一个真实 decode 步恰好被 `_trace_qfa_plan` 里的
  `metadata[:4].tolist()`（D2H）意外串行化了——探针每 shape 只打 3 次，之后窗口就打开。
  ⇒ **验证这个修复必须跑足够多的 decode 步**，别被探针的前 3 次掩盖。
- **MTP 的 draft 层必须也走 QFA**（2026-09-01 实测，已修 `81fe7b114`）：`_qfa_serves` 原本无条件
  `not _EXTRA_CTX.is_draft_model`，于是 draft 落到他们的 FIA 路径，D=256 直接 EZ0010 崩。
  那条排除只对**图捕获**成立（MTP 多步合并进一张图 + `full_graph_qfa` 用的是 target 的
  graph params），eager 下每个 draft 步各带自己的 metadata，没有这个问题。已把排除挪到
  `forward()` 里、只在 `qfa_capture` 时生效。
- ✅ **第一个端到端可用配置（2026-09-01 实机）**：`MTP=3 GRAPH=1 QFA=1
  CUDAGRAPH_MODE=PIECEWISE`，权重 Qwen3.5-35B-A3B-mxfp4-c8，**可服务**。
  QFA 覆盖 target 与 draft 的 prefill + decode，KV cache 为 MXFP8，MTP 开着。
  代价是 attention 整段留在图外（放弃 full-graph 收益）。
  ⚠️ 精度仍不可谈——V scale 还没读进来，走的是 127 兜底。
- **`81fe7b114` 只把 draft 换了一半，别读成「MTP 全面换成 QFA」**：排除条件从「draft 永远
  不走 QFA」收窄成「**draft 在 FULL 捕获时**不走 QFA」（`attention_v1.py:3146` 的
  `not (qfa_capture and _EXTRA_CTX.is_draft_model)`）。PIECEWISE/eager 下 draft 走 QFA
  （MTP+PIECEWISE 因此能跑）；**FULL 捕获下 draft 仍走 FIA v2** ⇒ D=256 崩。
  第四格故意留着：`full_graph_qfa` 没为 draft 写过（`:206` 对 draft 直接 return、
  `:1971` 取的是 target 的 graph params、`qfa_buffers` key 里没有 draft 步），
  **完全去掉排除会从「报错」变成「死锁」**——事件挂进 target 字典，永远没人 record。
  ~~⇒ 即使全图捕获修好，`FULL + MTP` 仍起不来~~ **这条推论错了，已实测推翻**：
  「多步合并进一张图」不是障碍，展平 `(draft_step, key)` 就能对齐；`9633c8ac2` 之后
  `MTP=3 GRAPH=1 QFA=1` 可服务。
- **bf16 的 MTP 是怎么进图的（要给 QFA 补 draft 支持就照抄这个）**：draft 模型自己被
  `ACLGraphWrapper` 包着（`llm_base_proposer.py:539`），有独立 `_draft_graph_params`；
  `full_graph_fia`（`attention_v1.py:1328`）按 `is_draft_model` 选 `get_draft_graph_params()`，
  N 个 draft 步 × L 层的 FIA 算子按顺序捕获进**同一张图**；replay 时 `update_graph_params`
  的 draft 分支用 `draft_attn_metadatas`（每步一份），把 `(draft_step, key)` **展平成有序列表**，
  按捕获顺序 zip 上 `attn_params/handles/events` 逐个 `graph_task_update`。
  ⇒ **「多步合并进一张图」不是无解**，FIA 就是靠展平顺序 + 逐个重绑解决的。
  QFA 的对应改动：`qfa_buffers`/`qfa_events` 改按 `(num_tokens, draft_step)` 建，
  `_update_qfa_graph_buffers` 接收并 enumerate `draft_attn_metadatas` 逐步刷。
  QFA 不需要 `graph_task_update`（读捕获期自持 buffer），只是「刷 N 份而不是 1 份」。
- **MTP 层在 35B checkpoint 里完全没量化**：`mtp.layers.0.self_attn.*` 只有 6 个张量，
  连 `weight_scale` 都没有（yaml 的 `linear_quant` include 是 `*language*`，MTP 前缀是 `mtp.`
  被排除）。⇒ 启动日志里 `min=127 max=127 distinct=1` 那条就是它，**权重里本来就没有，
  加映射也没用**。draft 的 V cache 因此无缩放 ⇒ 掉接受率但不出错（target 会验证）。
- ~~**推论：在 C8-MXFP cache 上，D=256 的模型做不到「MTP + FULL 图」**~~
  **整条作废（2026-09-01 实测）**。它建立在「draft 的图只能捕获 FIA v2」之上，而 draft 现在
  走 QFA，根本不碰那条 D=64/128 的限制。`MTP=3 GRAPH=1 QFA=1` 可服务。
  ⇒ C8 下要 MTP **不必**退到 `PIECEWISE`。
  ⚠️ **这条只对 C8-MXFP 成立，别泛化成「D=256 不能 MTP+图」**：那条 D 限制是 MXFP8 全量化
  专属的（报错原文 "In the MXFP8 full quant scenario"），bf16 cache 下 draft 捕获的是普通
  `npu_fused_infer_attention_score`，没有这个限制——基线的 `MTP3+GRAPH1` 一直能服务。
- 🔻 **QFA 的 flash-decode 是「host/kernel/AICPU 三处一致的编译期关闭」，不是一行 `fdOn=0`
  的性能疏忽**（2026-09-03 静态读码，三处都能 file:line 复核；上游 master 同样未改）。
  这条**推翻**了此前「kernel 侧 FD 实现完整、`VecFdBlock` 被实例化进 Kernel 不是 Dummy、
  运行期由 plan 的 is_fd 决定、链路是通的」那个说法 —— **链路在编译期就断了**：
  ① **host tiling**：`quant_flash_attn_tiling_mxfp8.cpp:127`，`SplitPolicy()` 算完切分策略后
     无条件 `flashDecodeFlag_ = false;`（该函数正是本该判断要不要 FD 的地方，全文件再无别的赋值）。
     它同时驱动 `tilingKeyInfo_.isFd`（`:207`）和 FD 的 workspace 分配（`:252-258` 的
     `accumOutSize`/`logSumExpSize`，**代码写好了但永远进不去**）；
  ② **kernel**：`quant_flash_attn.cpp:59` 与 `:189` 两处 `constexpr bool isFdConst = false;`
     → `VecFaBlock` 的 `isFd` 模板参数 → `block_vec_mxfp8.h:77` `FLASH_DECODE = isFd`
     → `kernel_mxfp8.h:63` `FLASH_DECODE = VecFaBlockType::FLASH_DECODE` → 两个
     **`if constexpr (FLASH_DECODE)`**（`:204` 初始化 `vecFdBlock_`、`:808` 调 `FlashDecode()`）
     **被编译期整段剪掉**。`VecFdBlock` 确实作为模板实参传进了 Kernel，但它的**所有调用点都在
     `if constexpr(false)` 内** ⇒「实例化了所以链路通」是误读，模板实参 ≠ 代码被生成；
  ③ **AICPU**：`quant_flash_attn_metadata_aicpu.cpp:225` `param.fdOn = 0`，plan 不排 FD 任务。
  三者**互相配套且自洽**（tilingKey.isFd=0 选中的正是 isFdConst=false 那个 kernel 实例），
  ⇒ 这是**有意关闭**（功能位全部预留、逻辑未实现），不是遗漏。`load_balance_common.h:59`
  的默认 `fdOn{true}` 是 common 库给别的算子用的，不能当作「QFA 本该开着」的证据。
  ⛔ **因此「把 `fdOn` 改成 1 重编来自证」这个实验不能做，做了必错**：只改 AICPU 会让 plan
  排出 FD 任务（`ScheduleFd` 填 fdTaskNum / workspaceIdx / AIV 区），而 kernel 里没有任何
  代码消费它们、workspace 也没分配 ⇒ 拿不到加速，只会得到错误结果或非法访问。真要开需要
  同时动三处（含让 tilingKey 能选到 `isFdConst=true` 的 kernel 实例），那是**改算子本体**，
  硬约束禁止，工作量也是算子团队级别的。
- **QFA vs FIA 单算子实测（2026-09-03，35B 形状，`test_qfa_as_fia_npu.py --model 35b --all`）**：
  **prefill 越长越赢、decode 全线落败**。prefill 512/1594/4k/16k = 1.40x/1.66x/2.32x/2.45x
  （fp8 cube 吞吐是 bf16 两倍，2.45x = 2x + 带宽收益）；decode b32-1k/4k/16k =
  0.55x/0.51x/**0.24x**、b8-32k 0.47x，**唯独 b1-128k 是 1.47x**。
  根因就是上面那条 FD 关闭：没有 FD，decode 并行度只能来自 `batch × heads`、每核串行扫完
  整段 KV；batch=1 时两边并行度都受限，QFA 读一半字节的带宽优势才显出来。
  精度（三方对比，拆开量化损失与算子差异）：QFA vs FIA(bf16) 5.30%/5.69%（dense/paged）、
  vs FIA(反量化) 2.65%/2.79%；27B 形状给出几乎一样的数（5.37/5.69、2.69/2.72）
  ⇒ 是算子固有特性。分解：`sqrt(5.30² − 2.65²) ≈ 4.6%` 是 K/V 压 MXFP8 的信息损失，
  2.65% 是 fp8 matmul 相对 bf16 的计算精度差。
  ⚠️ **5.3% 是下界**：脚本用 `quant_v_by_sequence` 现算的**最优** scale，框架用的是 checkpoint
  里的**静态** per-channel scale，只会更差；bench 传的 `max_seqlen_kv` 是紧值而框架传 133120
  常量，框架的 tiling 只会更保守。
  换算端到端：35B 有 10 个全注意力层、meta 每步只付一次 ⇒ decode-b32-16k 每步 attention
  ≈ 10×2.853 + 0.059 ≈ **28.6 ms**，同场景 FIA 是 6.9 ms —— **但 FIA 在 C8 + D=256 下根本
  跑不起来（EZ0010）**，所以那不是可选项，是「本可以达到」的参照。
  另外两个同样指向并行度、但未验的点：`param.l2Byte = 0` 让 `sectionNum` 恒为 1；
  `blockDim=32` 而 plan 按 `AIC_CORE_NUM=36` 排布（第 32–35 号核的分片没人执行，eager 下同样
  是 32，不是崩因但没查过）。
- **两份 35B 权重的 C8 形态实测（2026-09-03，`scripts/debug/check_c8_mxfp_weight_support.py`）**：
  ① `/mnt/share/weight/Qwen3.5-35B-A3B-mxfp4-c8` —— **可跑 C8**。`kv_cache_type` 正确，
     10 个全注意力层 `[3,7,11,15,19,23,27,31,35,39]` 全部带 `v_proj.v_scale`
     `[512,1]` uint8；零值通道 **131/5120 = 2.56%**，逐层 2~15 个，**layer 39 独占 84 个**；
     值域 0..122、distinct 6~9。MTP 层无 V scale。
  ② `/mnt/share/weight/qwen3_6_35b_a3b_w8a8_mxfp8_c8_fix_mtp` —— **QFA 读不了**，
     但**不是因为没做 KV cache 量化**（这一点第一版记错过，别再重复）：它做了，做的是
     **FAKQuant**（`fa_quant_type="FAKQuant"`，每层 `fa_k/fa_v.scale` + `.offset` 各 10 个），
     而 `kv_cache_type` 为空。两套配方的区别：MXFP8 的 E8M0 scale 是 2 的幂、**不需要 offset**，
     且配方是 `K_DYNAMIC` 只有 V 带静态 scale；FAKQuant 是 per-channel scale+offset、K/V 都带。
     **判据就看 offset 在不在。**
     框架侧链路（`modelslim_config.py` 读码确认）：`fa_quant_type` 非空 ⇒ `enable_fa_quant`；
     `get_quant_type_for_layer` 对 attention 层直接返回 `"FAKQuant"` ⇒ `get_scheme_class`
     选中 `kv_c8.py:30` 的 **`AscendFAQuantAttentionMethod`**；`get_quant_method` 里
     `is_fa_quant_layer` 那个 elif **排在 `enable_mxfp_c8_quant` 之前** ⇒ 10 层全部走它。
     `enabling_fa_quant` 在 **A5 上无条件启用**（非 A5 才要求 PD 分离的 decode 实例），
     `get_kv_quant_dtype` 在 A5 上给 `float8_e4m3fn`（非 A5 才是 int8），K/V 都量化。
     ⚠️ **`AscendFAQuantAttentionMethod` 是为 MLA 写的**（docstring 明写 MLA-based，
     `process_weights_after_loading` 里 `fa_k_scale.repeat(self.kv_lora_rank)` 用的是
     MLA 专属的 `kv_lora_rank`，非 MLA 取到 0）。Qwen3.5 是 GQA ⇒ 该行的安全与否取决于
     **每卡 `num_kv_heads`**：等于 1 时 `squeeze` 出 0-dim、`unsqueeze(0)` 得 1D，
     `repeat(0)` 合法（`quant_kscale` 成空张量，只要非 MLA 路径不读就无害）；
     **≥2 时是 2D，`repeat` 只收到 1 个参数会 RuntimeError**。Qwen3.5 只有 2 个 KV head，
     所以 TP≥2 大概率躲过去、TP=1 会炸。**未实机验证，别当定论。**
     它的 linear 配方也不同（`w8a8_mxfp8` vs 旧的 `mxfp4`），张量数 64013 vs 63481。
     ⇒ **不能测 QFA/C8 精度**；也不适合当 bf16 基线（既走 FAKQuant fp8 cache 而非 bf16，
     linear 配方又变了）。
     名字里的 `fix_mtp` 修的是「MTP 层 linear 完全没量化」那条，**已确证**：`mtp.` 张量
     1557 vs 旧 785，多出来的正是配对的 `weight_scale`（`mtp.fc.weight_scale`、每个 expert
     三个），旧权重一个都没有。但 KV cache 侧仍未覆盖 MTP：`k_proj.weight_scale` 有 11 个
     （含 MTP 层）而 `fa_k.scale` 只有 10 个（不含）。⇒ **`fix_mtp` 只修 linear，与 KV cache 无关。**
  ⚠️ 又一次印证 [`2.4t-weight-history.md`](./2.4t-weight-history.md) 的「**权重目录名不可信**」：这已经是第四次。
  要让某份权重支持 C8，得让 ModelSlim 按 `dynamic_cache`（scope per_channel / dtype mxfp8）
  对 KV cache 重量化，产出 `kv_cache_type` 与每层 `v_proj.v_scale`。
- **QFA dump/replay 已跑通（2026-09-04）**：插桩在 `AscendC8MXFPAttentionBackendImpl.forward`
  两处（量化后写 cache 前的 bf16 真值 + `_qfa_paged_call` 的输入输出），env 开关
  `VLLM_ASCEND_QFA_DUMP_DIR` / `VLLM_ASCEND_QFA_DUMP_CALLS`，分支 `feat-qfa-dump`。
  **只能 eager（GRAPH=0）**：捕获期 D2H 会 EE1016，replay 期又不跑 Python。
  10 层全落盘，`replay_qfa_dump.py` 回放 **10/10 bit-exact**，metadata 由 op_kwargs 重建后
  结果同样 bit-exact，plan header 恒 `[1, 0, 128, 256]`——**其中 isFd=0 是「FD 关闭」的运行期实证**。
  按 block 取 cache 必须走字节视图：`aclnnIndex` 不支持 FP8（EZ1001）。
- ⛔ **别再写「在 CPU 上造 golden 来拆量化/算子损失」这类脚本**（`analyze_qfa_dump.py` 已删）。
  要算对 golden 就得精确复刻算子的每一条内部约定，一天里连栽四次：参考喂了 bf16 的 q 而算子
  吃的是量化后的 q；`q_descale` 是 `float8_e8m0fnu` 而 cache 的 scale 是 `uint8`，对前者
  `.to(int64)` 是数值转换（2^-7→0）不是取指数字节；还剩 compute 32.5% 到删除时都没查清。
  自测只能证明「我的 dequant 与我的 quant 互逆」，**自洽但验不出整体偏移**。
  ⇒ 精度问题改由端到端评测回答（`C8=0` 开关在 `junlin-qfa-c8switch` 分支），
  算子级则用 `bench/test_qfa_vs_fia.py`（合成输入，判据独立）。
- 🔎 **一条未追完的线索：V 的量化误差可能远大于 K**。用框架自己存下的乘数
  （`v_cache_scale_float_reciprocal`）去预测框架自己写出的 `value_mxfp8`，rel_l2 **39%**；
  同一份 dump 里 K 和 q 的量化误差都只有 3.1~3.5%。这一步不经过任何自造 golden，
  所以不受上面那些 bug 影响。可能是缩放后大量撞上 e4m3 的 448 上限（饱和），也可能挤进
  subnormal 区（3 位尾数在那里几乎没有相对精度）。**16 token 的 prefill 会放大它**，
  要下结论得用几百 token 复测。与 layer 39 那 84/512 个零值通道大概率同源——
  都指向 checkpoint 的 V scale 校准。
- **AICPU 故障归因会骗人**：下发异步，traceback 指向当时在等的算子；定位靠在阶段之间
  插 `torch.npu.synchronize()`。`ASCEND_LAUNCH_BLOCKING=1` 也值得开。
- **attn_mask 语义**：`triu(2048², diagonal=1)` int8（1=屏蔽未来），与 QFA `mask_mode=3`
  一致，也正是 FIA 那边 `get_splitfuse_attn_mask()` 给的；官方 md 里 tril 的示例是错的，
  以官方**测试代码**为准。
- **数值判据**：QFA vs FIA 喂反量化后同输入 cos=0.9996（布局一致），喂原始 bf16
  rel_l2≈5.7%（MXFP8 本身损失）。度量必须 float64 累加，近千万元素在 float32 下余弦
  会漂到 1.0008。two-per-mille 那套判据是比 golden 用的，不适用于量化 vs 全精度。
- **`aten::scatter_reduce.two_out` 在 NPU 上没 kernel**，回退 CPU 并拖进设备同步，
  热路径要避开。
- **FIA V5 有同族 MxFP8**（scale 布局与 QFA 逐字段相同）但 HeadDim 仅 64/128
  （27B D=256 出局）、PA 仅 BnNBsD/NZ 无 BBND、且受 CANN 镜像版本挟持 —— 选 QFA 的硬理由。
  vLLM `--kv-cache-dtype fp8` 是静态 per-tensor 体系且 ascend 后端未落地，与 MXFP8 无关。
- **vendor 新算子进 csrc 的三个构建坑**（再 vendor 会复现）：
  ① `cmake/scripts/util/opdesc_parser.py` 的 SOC_TO_SHORT_SOC_MAP 缺 950 细分 die 名
  （ascend950pr_9589/9599）→ KeyError，补映射到 ascend950；
  ② csrc 不链 CANN opbase 库 → 引用 `Ops::Base::*` 在 libcust_opmaster_rt2.0.so 留
  undefined symbol，**TBE dlopen tiling so 失败会炸掉全仓所有 TILING_DATA_DEF 算子的注册**
  （"do not registe tiling struct" 批量报错）——解法是算子内补 compat .cpp 显式编入
  tiling obj；诊断用 `scripts/debug/diag_qfa_tiling_registry.sh`（ldd -r）；
  ③ 裸名 include（"util.h"）会被 CANN 内置同名头抢先解析（旧版枚举缺 LAYOUT_NTD →
  no member + 级联 unknown type）——vendor 时裸名必须归一为显式相对路径。
  `add_ops_compile_options` 注入的私有 -I 对 opc kernel 编译不可靠，别依赖。

**谁走 QFA、谁还是 FIA**（QFA=1）：**prefill、eager decode 与 FULL 图里的 decode 全部走 QFA**
（2026-09-01 起，见下面的里程碑）。~~aclgraph 里的 decode 仍然是 FIA~~ 这条已作废。
另外恒走 FIA 的：MTP draft 模型的图（多步合并进同一张图，捕获出来的 op 无法判断该读哪一步的
长度）、有 sinks / 滑窗 / 非 causal 的层。判据见 `_qfa_serves`。

**量化必须入图，这是硬约束**：FULL_DECODE_ONLY 捕获整个模型前向，q 由图内 QKV 投影
产出；本步 K/V 由图内 `reshape_and_cache` 落进 cache，图外先量化会漏掉当前 token。
两者都要等 MXFP8 cache 才能搬出去。真捕获不了的话短期退路只有：decode 退回 FIA 图，
或把 `cudagraph_mode` 换成 `PIECEWISE` 让 attention 整段留在图外（纯配置改动）。

**图路径风险已实测（2026-08-30，`scripts/debug/test_qfa_graph_capture_npu.py`，四个 case
各独立子进程）**：
- ✅ **`npu_dynamic_mx_quant` 可以在 aclgraph capture 内捕获**，且 replay 时是真的重读
  cache（换掉整份 bf16 cache 内容与 kv 长度后 replay 仍与 eager 逐字节相同）。
- ✅ **`max_seqlen_kv=133120` 常量下 plan 仍是 4096 int32**，输出与 2048 常量一致。
- ✅ **占位 `v_descale` 与真的等价**（输出 bit-exact）。注意 plan 字节**不能拿来比**：
  算子输出是 `at::empty` 分配的，只写头部与 sectionNum 个 section，尾部是分配器残留——
  同参数、不同 buffer 的两次调用就差 ~2782/4096 个 int32。比之前必须让三份 plan
  同时持有，否则先后分配会复用同一块内存、假装一致。
- ~~🔴 **QFA 进图必崩**~~ **已解决（2026-09-01），解法见里程碑那条**。以下是解决前的排查记录，
  结论「共同因子是 QFA 有没有真的进图」仍然成立，只是崩因已找到：
  （2026-08-31 单变量实验坐实）。三次并排：
  常量全局+捕获不限状态（208913440）崩；常量限 decode+捕获不限状态崩；常量限 decode+
  捕获限 decode（5f2b644ca）正常——**共同因子是「QFA 有没有真的进图」，不是常量**。
  5f2b644ca 让服务活过来是因为把功能关掉了：日志实证 target 的图以
  `attn_state=ChunkedPrefill` 捕获（不是 DecodeOnly），而 `_qfa_serves` 要求
  DecodeOnly/SpecDecoding ⇒ 对 target 恒 False，QFA 从未进图，decode 图里还是 FIA。
  已排除：① 常量本身（PREFILL-MAXKV 绿）；② 「找不到 .qfa 静默返回读未初始化 buffer」
  （日志 `update tokens=4 -> refreshing {'events': 16}`，刷新成功、形状全对）；③ 环境波动；
  ④ **引擎结构本身**——`ENGINE-INPOOL` / `ENGINE-OUTPOOL` 按 `full_graph_qfa` 同构复现
  （16 层共用一套 capture-owned buffer、多个图尺寸共享一个 graph pool、事件一次性 record、
  replay 前刷 plan），逐层 bit-exact 全绿，缓冲区在池内池外都绿。
  **⇒ 我能指名道姓的假设已经用完。下一步不再靠猜：AI Core 报错本身会打 `fault kernel_name`，
  直接说明炸的是 QuantFlashAttn、图内量化还是 ReshapeAndCache。这一行从头到尾没人看过。**
- 🔴 **`metadata` 是必需参数，文档说「可选」是错的**（2026-08-31 实测）：传 None 直接
  `Invalid_Input(EZ0004): Parameter metadata of QuantFlashAttn is required`，dense/paged 都拒。
  ⇒ 无法靠丢掉 plan 来绕开文档那条「两次调用入参必须一致，否则未定义行为（精度问题、
  非法内存访问等）」的约束。
- ~~🔴 **QFA 没有 out 变体 ⇒ graph_task_update 不可用**~~ **说法本身就错，是我们自己的
  接入疏漏（2026-09-01 认清）**：不是「QFA 没有」，是**我们 vendor 时没写**——
  `ce66a81a0` 的 binding 只出了 `at::empty` 那个分配式重载。**aclnn 一直是 out 语义**，`aclnnQuantFlashAttnGetWorkspaceSize`
  的倒数第三、四个参数就是 `attnOut` / `softmaxLseOptional`，与
  `aclnnFusedInferAttentionScoreGetWorkspaceSize` 的 `attentionOut` / `softmaxLse` 同构。
  分配式重载只是用 `at::empty` 把这件事藏起来了。**加一个 `.out` overload 是纯 binding 层
  的活，不碰 kernel**，`e3223340f` 做的就是这个（adpt.h 加薄封装 + torch_binding.cpp 加
  schema + meta 实现）。文档确实通篇没提 aclgraph——但那是因为算子文档本来就不讲 PTA API。
- 🔑 **捕获期与运行期的 `attn_state` 本来就不同**（源码印证，非推断）：MTP + 非 MLA 时
  `_dummy_run` 捕获用 `ChunkedPrefill`（model_runner_v1.py:3516-3522，注释写明
  `SpecDecoding is only designed for mla`），而真实 decode 步 `_build_attn_state` 判
  `SpecDecoding`（同文件 1546-1551）。⇒ **任何跨捕获/replay 依赖 attn_state 的逻辑都是错的**。
  我的 `_QFA_CAPTURED_STATES` 设计从根上踩了这条：`_qfa_serves` 用它判断导致 QFA 永不进图；
  `_attach_qfa_inputs` 用它选 max_seqlen_kv 会让捕获期烘进紧值、replay 期 plan 用 133120，两者不一致。
- `PREFILL-MAXKV` case 已验证：prefill 形状（B=1 q=1594 kv=1594）下 `max_seqlen_kv=133120`
  与紧值 bit-exact，常量本身在 prefill 下是安全的。
- ⚠️ **测试台自身的坑（踩过一次）**：`stream = torch_npu.npu.current_stream()` 必须在
  `with torch.npu.graph(...)` **块内**取。`torch.npu.graph` 在侧流上捕获，块外取到的是默认流，
  `event.wait` 于是真的挂死默认流，下一次捕获的隐式 synchronize 直接死锁——而且此前几个
  绿灯里事件机制**根本没进过图**。引擎不吃这个亏是因为它在被捕获的前向里取流。
- 显存：`memory_reserved` 的增量**量不到东西**（64 块与 9672 块都报 +38MiB，因为 eager
  预热已经把段 reserve 好了）；要看 `max_memory_allocated` 的增量。
⚠️ `diag_qfa_metadata_size.py` **量不了这些**——它是为已废弃的五平面设计写的，扫的是
v_scale 平面在单 buffer 里的 int32 偏移边界。

- **捕获块内 `torch.zeros` 会把清零本身录成图节点**（2026-09-01 实测踩过）：每次 replay
  先把缓冲区抹成 0 再拿它用，replay 前刷进去的内容全丢。症状是图内所有写落到 slot 0。
  `torch.empty` 不录 kernel。`QFAGraphBuffers.like` 用 `empty_like` 是对的——照抄时
  **别"顺手改成 zeros 更安全"**。
- **`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 显著改变图池地址布局**：同样 4 个
  捕获尺寸，不开跨 57MB，开了只有 4MB。27B.sh 导出的就是它 ⇒ 任何比对池地址的实验
  必须对齐这个变量，否则地址没有可比性。
- 🔴 **小脚本在真机形状 + 引擎结构下复现不了全图崩溃**（2026-09-01，
  `scripts/debug/test_qfa_fullgraph_repro_npu.py`）：35B 真形状（16 heads / 2 kv /
  D=256 / block 512 / 1612 blocks / max_seqlen_kv=133120 / 块表 260 列 / 10 层）、
  引擎结构（捕获期自持 buffer + 多尺寸共池 + 事件一次性 record + replay 前刷 plan）、
  33 个捕获尺寸共池、**图内写 cache 再读同一 cache**——`REAL` / `SIZES-ALL` /
  `INGRAPH-CACHE` / `WRITE-IDEMPOTENT` **全绿，逐层 bit-exact**。
  ⇒ 崩溃需要的条件不在这些维度里。池跨度实测 4 尺寸 4MB、33 尺寸 163MB，而真机故障
  地址距最远 buffer ~248MB，规模仍差一个量级。
  ⚠️ **旧的 `ENGINE-INPOOL`/`ENGINE-OUTPOOL` 绿灯不适用于当前代码**：它们 lift 的
  `_qfa_quant`/`_qfa_quant_v` 在 `c85c545ec` 已随 bf16 路径删除，
  `test_qfa_graph_capture_npu.py` 在 junlin-qfa-graph 上连 bootstrap 都过不去；而且跑的是
  max_seqlen_kv=2048 / block 128 / kv_heads 4 / 块表 3 列，没一样是真机的。
  「引擎结构已排除」这条要按新脚本重新表述，别再引旧证据。
- **`copy_between_host_and_device` 这个函数名在两种完全不同的故障里都会出现**，只看错误码：
  `107030`/`EE1016 ... during the capture stage is not supported` = 捕获期有人做了 D2H，
  图没建成；`507011`/`the model stream execute failed` = 图建成了、replay 时算子炸。
  交接文档记的是后者。junlin-qfa-graph 的捕获路径（`full_graph_qfa` / `_qfa_paged_call` /
  `_qfa_quant_q`）静态扫过**没有 D2H**，仅有的两处在 builder 侧捕获块外
  （`attention_v1.py:126` 探针 `metadata[:4].tolist()`、`:631` 的 `torch.tensor(...)`）。

- **`out` 变体是为 aclgraph 专门做的适配，不是普遍能力**（2026-09-01 实机反射确认）：
  torch_npu 里 49 个 attention 相关算子只有 4 个有 `out` overload——
  `npu_fused_infer_attention_score`、`_v2`（正是 vllm-ascend 图捕获用的那两个）、
  `npu_fusion_attention_v3`、`_grad_v3`（训练侧）。**IFA、PFA、5 个
  `_npu_paged_attention*` 全都没有**（后者连 OpOverloadPacket 都不是）。
  ⇒ 「QFA 没有 out 变体」不是我们用错 API，是这项适配没人给它做。提需求时可对标 FIA。
- **FIA 的 replay 重绑是「整个调用重跑一遍」，不是「重绑张量地址」**（读码确认，之前的
  表述不准）：`attention_v1.py:1288` 的 `graph_task_update_begin(update_stream, handle)`
  之后，把完整参数表的 `npu_fused_infer_attention_score.out(...)` 原样再调一次，
  再 `graph_task_update_end`。aclnn 每次调用都在 host 侧算一次 tiling ⇒
  **FIA 每次 replay 都拿到新算的 tiling 并写回图里那个 task，QFA 一次都不刷。**
  捕获侧对应 `:1472` 的 `graph_task_group_begin` → `.out(...)` → `:1496`
  `graph_task_group_end` 拿 handle。三个必要条件（out 变体 → task group → task
  update）QFA 一个都不满足。
  ⇒ CANN 的设计预期就是「被捕获的算子要靠 update 才能正确 replay」。
- **文档层次别找错**（问过一次）：`ops-transformer` 仓只有 kernel 实现 + README
  （算子语义/参数/约束），**不含任何 PTA API**，所以 `quant_flash_attn.md`「通篇没提
  aclgraph」是意料之中，那本来就不归它讲。`.out()` / `graph_task_*` 属于 torch_npu
  的 Python 绑定层，要在 torch_npu 的 API 文档或 op-plugin 的 yaml 里找。
- **真机单变量又排除两条**（2026-09-01）：`MAX_MODEL_LEN=8192` 照崩 ⇒ 烘死的
  `max_seqlen_kv` 常量出局；`MAX_NUM_SEQS=1 CAPTURE_SIZES=1` 照崩 ⇒ batch 大小、
  plan 的每核切分规模、多图共享池全部出局。27B.sh 已支持这两个开关。
- **故障地址是彻底的野指针，不是越界一点**（2026-09-01 拿到完整 AIC_INFO 表）：
  17 个入参地址与 dump size 逐个核对**全部正确**（k/v cache 各 422576128 B、两个 scale
  平面各 13205504 B、q_fp8 4096 B、q_descale 128 B）。而各核故障地址
  （`SU_ERR_INFO_T0_2` 低 16 位 `409c` 拼 `SU_ERR_INFO_T0_1`）落在池基址
  **+179MB~748MB**，池里所有合法入参却都在基址 **+64KB 以内**（最远 attn_out `+0xfa00`）。
  逐核不同、位模式像随机数 ⇒ kernel 在拿垃圾当切分参数。
  `tiling` 是 **host 地址**（`0x12004028b0a8`），args 里有一个字是 `0xa5a5a5a5<junk>`
  ——已释放内存的 poison pattern。

- **vendor 新算子时，binding 要主动出 out 变体**——这条已写进
  `docs/vendor-ops-transformer-op.md` 的「坑五」，连同图捕获该怎么按 task update 写、
  draft 怎么展平对齐、怎么用日志证明算子真进了图。再 vendor 算子时先读那一节。
  教训本身：把自己造成的限制（binding 少写一个重载）误判成算子能力缺失，据此推出
  「QFA 可能不支持 aclgraph」，还差点整理成结论去问算子团队。**下次遇到「这个算子做不到 X」
  的结论，先确认 X 需要的接口我们自己有没有暴露全。**

- ✅ **官方有 aclgraph 测试，进图不是我们自创的**（2026-09-03 查实，推翻此前「无官方背书」
  的说法）：`ops-transformer/attention/quant_flash_attn/tests/assets/impl/graph.py` 里的
  `QuantFlashAttnAclGraph`（418 行），docstring 明写「metadata 构建在 capture 之前」、
  「forward 只调主算子」——**与我们的设计一致**。还提到 `SparseFlashMlaAclGraph`，说明这是
  跨算子的既有模式。差别只在官方 forward 用返回值形式而非 `.out()`：golden 测试每次 replay
  喂同样输入，不需要 `graph_task_update`；真实服务每步 metadata/长度都变才需要。
  ⇒ 文档没提 aclgraph 不等于不支持，**下次别从「文档没写」推「不支持」**。
- **ops-transformer 上游状态（2026-09-03 复核）**：本地克隆停在 vendor 的 `14cf794f3`，落后
  master **217 个提交**，其中 10 个动了 QFA。要点：
  ① **FD 三处开关在 master 上一字未改**（见下面那条硬事实）⇒ **rebase vendor 解决不了
     decode 性能问题**，③ 这条路的价值只剩「跟上 breaking change + 为换官方包趟路」；
  ② `fb89a5e6a` **metadata 删除 v_descale 参数、新增 head_dim_v** —— breaking change，
     我们还在传 `_qfa_v_descale_stub`，rebase vendor 或换官方包时必须改。顺带印证了
     「v_descale 必须非空但 PA_BBND 下没人读」那条判断，官方直接把它删了；
  ③ ~~`a006bbc06` 的 BBND 拦截校验值得看~~ **已看，与我们无关（2026-09-03 读完）**：
     那 5 行只是给 PA_BBND 补一条「vDescale 必须 5D」的秩检查，而我们传的 stub 本来就是
     5D ⇒ 拦不到我们，反而印证 5D 占位是对的规格。同提交标题的另一半「修复 QFA graph 路径下
     PA 初始化问题」改的**全在 `tests/assets/`**（graph.py 等三个文件各 +4 个 layout 参数
     LAYOUT_Q/LAYOUT_Q_DESCALE/LAYOUT_KV/LAYOUT_OUT），是他们**测试台**此前把四个 layout
     当同一个值传、PA 场景的 graph 测试压根初始化不了 —— 测试台的 bug，与框架侧接入无关。
     ⚠️ 旁证：官方的 QFA graph 测试台在 2026-08-29 之前**没能力测 PA layout**
     ⇒「QFA + PA_BBND + aclgraph」这个组合官方也是刚开始覆盖，我们踩的是新地；
  ④ `c73726cc0`「AclGraph batch size golden」只动 HIF8 测试脚本，与我们无关。

**现役脚本**（2026-09-04 按存活周期重组，旧路径全部失效）：
`scripts/27B.sh`（QFA/GRAPH/MTP/C8/CAPTURE_SIZES/CUDAGRAPH_MODE 开关）、`scripts/curl.sh`；
`scripts/bench/`——`test_qfa_op.py`（原 test_junlin_qfa_npu，8 case 对 golden）、
`test_qfa_vs_fia.py`（原 test_qfa_as_fia_npu，三方精度 + `--bench` 性能）、
`replay_qfa_dump.py`（把真机 dump 喂回算子，验输入自洽 + 输出 bit-exact）；
`scripts/checks/`——`c8_mxfp_weight_support.py`（权重能否跑 C8 + V scale 零值统计）、
`compare_checkpoint_shapes.py`、`estimate_hbm_budget.py`、`chat_template_thinking.py`、
`probe_npu_memory.py`；`scripts/setup/`——`{build_qfa_ops,pip_install_qfa,diag_qfa_tiling_registry,
install-vllm-ascend,create-container}.sh`。`build_qfa_ops.sh` 装完 `_cann_ops_custom`
只剩 QFA，**该状态禁止起服务**。
⛔ **已删（别再找）**：`bench_qfa_vs_fia.py`（C8 后两个权重都跑不出有效对比）、
`test_qfa_fullgraph_repro_npu.py`（全图崩溃 09-01 已修）、`diag_qfa_metadata_size.py`（面向
已废弃的五平面 cache）、`run_doc_examples_qfa_npu.py`（从未跑过）、`check_moe_expert_shapes.py`
/ `verify_expert_split_axis.py` / `check_quant_desc_qwen35_moe_text.py`（2.4T 一次性排查，已结案）、
`analyze_qfa_dump.py`（见下）。

**`junlin-qfa-graph` 上已做的**（2026-08-31）：删掉 `_QFA_CAPTURED_STATES`——它建立在
「捕获期 attn_state 能代表运行期」这个假前提上（见 🔑）。同时 `max_seqlen_kv` 改成对所有
分页步都用常量（PrefillNoCache 在更早处已返回；prefill 长度 q 配常量由 PREFILL-MAXKV 实测安全）。
改完 QFA 才会真的进图，也才复现得了崩溃。
另加 `370c457d0`（2026-08-31）：`_trace_qfa_plan` 在 builder 里打印 AICPU plan 的 header
（`[0]` sectionNum / `[1]` isFd / `[2]` mBaseSize / `[3]` s2BaseSize）连同 batch /
max_seqlen_q / max_seqlen_kv / 四个张量形状，每个 (batch, max_q) 最多 3 行；
`_update_qfa_graph_buffers` 的 refreshing 行也带上四个 `copy_` 的 (缓冲区, 源) 形状对。
验的是一条从没测过的前提——`_attach_qfa_inputs` 注释里那句「max_seqlen_q 在捕获步上等于图尺寸，
hence constant」。mBaseSize/s2BaseSize 由 `AdjustSinnerAndSouter(head_dim, max_q, max_kv,
mask_mode, ...)` 算出，主算子捕获时把同一套 tiling 烘死，而 replay 只刷 plan 不刷 tiling
⇒ 两者只在这个 header 相同时才自洽。捕获期的行落在 vLLM 图捕获完成日志之前，replay 的在之后，
按日志顺序对比即可。

**MXFP8 KV cache 已合入（2026-08-31，`junlin-qfa-graph`）**，来源是同事的
`Tame21/vllm-ascend` 分支 `cache-c8`（10 个提交，从 upstream `3c85ee6fc` 分出，落后我们基线
235 个提交；squash 成一个提交后 rebase 到当前树，只冲突 8 处）。

- **V 的 scale 沿序列分组那个时间依赖，不用解了**：他们让 V 根本不做动态量化
  （`kv_cache_type=K_DYNAMIC_V_STATIC_MXFP8_PER_CHANNEL`），scale 从 ModelSlim checkpoint
  的 `v_proj.kv_cache_scale` 静态加载，一次广播进 v_scale_cache 就再不写（`save_v_scale_flag`）。
  K 仍逐 token 动态 mx 量化。⇒ 记忆里那三条路（bf16 底稿 +1.6% / 当前窗口 −24% / clamp −48%）作废。
- **框架侧怎么做到的**：`FullAttentionSpec.head_size` 撑成 `head_size + head_size//32`
  骗过 vLLM 的块预算，再把 raw buffer 切成 k / v / k_scale / v_scale 四份；hybrid
  （Mamba+attention 共享 padded buffer）走 `_split_hybrid_c8_mxfp_cache_buffer`。
  代码在 `vllm_ascend/device/mxfp_kv_cache.py` + `model_runner_v1.py` +
  `quantization/methods/kv_cache/mxfp_c8.py`。
- **我们的适配（现 `c85c545ec`）**：scale cache 改成 PA_BBND 轴序
  （K `(nb, bs, N, D//64, 2)`、V `(nb, bs//64, N, D, 2)`），QFA 直读，不量化不转置；
  `_qfa_paged_call` 只量化 q；`_qfa_quant_v` 与整条 bf16 QFA 路径删除；
  `VLLM_ASCEND_ENABLE_QFA` 现在还要 `is_c8_mxfp_kv_quant()` 才生效。
- **他们的 FIA 路径每步 `transpose(1,2).contiguous()` 整份 cache**（每层每步一次），
  因为 FIA 要 heads 在前。QFA 路径完全跳过它——这是选 QFA 的额外理由，不只是 D=256。
- **他们的 FIA 图路径不支持 MTP**（`_full_graph_mxfp8_decode` 有
  `len(actual_seq_qlen) != num_tokens` 就抛），且捕获限 DecodeOnly。我们的 QFA 捕获
  两条都不设限，MTP 仍在尝试范围内。

**三个未验证的前置**（实机第一轮就会暴露）：① checkpoint 必须带
`kv_cache_type=K_DYNAMIC_V_STATIC_MXFP8_PER_CHANNEL` 与每层 `v_proj.kv_cache_scale`，
否则 `mxfp_c8.py` 用 127 兜底（scale=1.0，等于 V 不量化）；② `refresh_block_size` 在
C8-MXFP 下强制 `block_size=512` 并**提前 return，跳过了 hybrid 的 model-specific 分支**
——Qwen3.8 正是 hybrid，启动异常先查这里；③ `GPU_MEM_UTIL=0.85` 是为已删除的每步量化设的，
现在可能不再需要，重新量。

`scripts/bench_qfa_vs_fia.py` 到这时才有意义。

⚠️ 别改 `csrc/attention/quant_flash_attn/` 下的 `op_api/` `op_host/` `op_kernel/` `common/`
（原样 vendor 的算子本体）；`quant_flash_attn_torch_adpt.h` 才是自己的 binding 适配层。

相关 [`2.4t-weight-history.md`](./2.4t-weight-history.md)、技能 `vendor-ascend-op` 的「数据搬运」、`AGENTS.md` 的「提交规范」。
