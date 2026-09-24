# docs —— 文档总目录

顶层三篇是"怎么做事"，`reference/` 是"反复要查的事实"，`cases/` 是"某次排查到底
怎么走的"。分界线是**寿命**：

- `/reference/` 里的表格与接口语义不随排查进展变形，换权重、换 CANN、rebase vendor
  之后仍然拿来对；
- `/cases/` 里的记录带日期、带当时的结论，**结案了也不删**——它记着哪些假设被推翻过、
  哪条路走死过，这是防止后人重走一遍的唯一来源。每篇文件头都写明状态
  （在途 / 已结案 / 部分过期），照状态判断哪些结论还能引用。

## 怎么做事

| 文档 | 内容 |
| --- | --- |
| `../AGENTS.md`（`../CLAUDE.md` 是它的符号链接） | 仓库硬性约束、worktree 工作流、提交规范、vllm-ascend 架构要点 |
| `../scripts/README.md` | 每个服务、诊断、压测、回归脚本的用途与运行方式 |
| `../scripts/local/README.md` | 本机（无 NPU）验证环境：venv lane、能拦住什么、已知短板 |
| `vendor-ops-transformer-op.md` | 以 QFA 为样板，把 ops-transformer 算子接进 `vllm-ascend/csrc` 的完整六道门，含 FIA 差异清单 |
| `qfa-metadata-capacity-bug.md` | 提给 cann/ops-transformer 的问题单：QFA metadata 在 prefill 下越界写、AICPU abort |

## reference —— 反复要查的事实

| 文档 | 内容 |
| --- | --- |
| `reference/qwen38-model-shapes.md` | 各 Qwen3.8 / Qwen3.5 尺寸的层数、头数、`model_type`、全注意力间隔；本机没权重也能拿结构 |
| `reference/qfa-op-contract.md` | QFA 两算子的接口契约：descale shape 表、`max_seqlen_*` 语义、tilingKey 六维各自能不能动、metadata plan 的共享边界 |
| `reference/c8-qfa-config-matrix.md` | 哪些 KV cache / attention 组合真能起服务，以及每条路被什么挡住 |

## cases —— 一次排查的完整记录

| 文档 | 状态 | 内容 |
| --- | --- | --- |
| `cases/397b-moe-ep-graph-bug.md` | 已结案（2026-09-09） | 397B 长 prompt 只吐一个 EOS：MoE 的 `_fused_output_is_reduced` 在编译区被烘死 |
| `cases/mtp-accept-graph.md` | 已结案（2026-09-09） | 一开 draft 图 MTP 接受率就塌：V scale 缓存在捕获期被误标为已填充 |
| `cases/qfa-pre-ops.md` | 部分真机待验 | QFA 前那 15 个小算子的归属、为什么不能提前到 `prepare_inputs`、已做的削减 |
| `cases/hybrid-c8-kv-capacity.md` | 本机已验、真机待验 | hybrid C8 的 KV 容量只有 BF16 四分之一的根因与验收预期值 |
| `cases/2.4t-weight-history.md` | 已结案（2026-08-29） | 2.4T 权重三次踩坑史，第三次是 routed expert 的 gate/up 沿 hidden 轴切错 |
| `cases/qfa-integration.md` | 部分过期（见文件头） | QFA 从 vendor 到进图的全过程，含已作废的里程碑与九次逐位 no-op 的教训 |

`graph-capture-timing`、`npu-remote-diagnose`、`vendor-ascend-op` 三个技能的结论就是从
上面这些 case 里提炼的——**要看机制与做法去技能，要看"当时为什么这么判断"来 cases**。
