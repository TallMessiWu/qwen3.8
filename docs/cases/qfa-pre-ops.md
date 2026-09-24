# QFA 前面那串小算子：归属、能不能提前、削到哪一步

> **状态**：部分真机待验——占位符视图那三项真机全绿，**gsm8k 精度结果当时未回填**。
> **来源**：2026-09 的 Claude Code 会话记忆（`~/.claude/projects/…/memory/`），2026-09-24 迁入本仓。
> ⚠️ 汇报时先把这句说在前面：**这次没有任何东西被提前到 `prepare_inputs`**，
> 「砍掉小算子」不等于「挪到 `prepare_inputs`」。

**背景**（2026-09-21）：profiling 里 `aclnnScatterPaKvCache` 之后、`QuantFlashAttnMetadata` 之前
有 15 个小算子，同事建议「提前到 prepare_inputs」。

**归属**：这个窗口每步只出现一次（metadata 算子每步只跑一次，窗口以它结尾）。
- 每步 1 次、挂在 `attn_metadata.qfa_metadata_cache` 上跨层共享：slot 拆解（cast/ge/zeros/where/
  floordiv×2/rem×2）、长度推导（clamp/cummax/clamp）、`v_descale` 占位符的 zeros。
- **每层 1 次**：K scale 的 Index → SWhere → IndexPutImpl（读-改-写，只为让 slot=-1 的行变空操作）。
  2.4T 是 23 层 × 3 = 69 个 kernel/步，这才是该砍的。

**为什么不能提前到 prepare_inputs**（分阶段）：
- `slot_mapping` 是 `_prepare_inputs` 里设备侧 Triton kernel 算的（`block_table.py`），`seq_lens`
  也是设备侧加出来的；异步调度 + MTP 下 CPU 只有 `optimistic_seq_lens_cpu`（乐观上界）→ **decode
  阶段没法在 CPU 上算**。
- 放到图外用设备 eager 算：FULL 图 replay 时图内 kernel 的 host 开销是 0，挪出去每步多付 ~9 次
  host 下发，还要写进持久 buffer 保证地址稳定 → 倒退。
- prefill（eager）能挪，但本来就每步 1 次，kernel 数不变，无收益。
- PR 自己的历史：长度推导原先就是 Python 侧 staging，因「捕获期冻住 + 异步调度下 pinned buffer
  竞争」才改成图内推导。

**已做**（`junlin-c8-mxfp-16614`，`8ac7268a4` + `efe299d67`，仅本机验证）：
- 每层 3 → 1：padding 行 clamp 到 slot 0 直接写。依据：block 0 是 vLLM BlockPool 保留的 null block
  （两个版本 `block_pool.py:190`），永不分给真实请求。prefill 无 -1 行 → 新旧逐字节一致；decode 图
  replay / MTP verify / draft 有 -1 行 → 差异只在 null slot（5080 上并行 scatter 验证过）。
- 每步 8 → 6（clamp 取代 ge+zeros_like+where）、1 → 0（占位符改用本层 V scale cache 头两字节的视图）。

**真机结果**（2026-09-21）：占位符视图的核对脚本三项全绿（算子接受该视图，plan 与 zeros stub
逐字节一致），脚本已删（主仓 `92819e6`）。gsm8k 精度当时在跑，结果未回填——下次先问。

**注意口径**：这次**没有任何东西被提前到 prepare_inputs**。用户/同事容易把「砍掉小算子」理解成
「挪到 prepare_inputs」，汇报时要先把这句说在前面。

**How to apply:** 判断「能不能跨层共享 / 能不能提前」必须按 prefill、decode 图 replay、MTP verify、
draft 分别过一遍；metadata 对象每步经 `build()` 新建（`build_for_cudagraph_capture` /
`build_for_drafting` 基类都直接调 `build()`），所以挂在它上面的每步缓存在各路径都不会跨步串用。
相关：[`qfa-integration.md`](./qfa-integration.md)、[`mtp-accept-graph.md`](./mtp-accept-graph.md)、[`hybrid-c8-kv-capacity.md`](./hybrid-c8-kv-capacity.md)。
