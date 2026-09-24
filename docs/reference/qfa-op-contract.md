# QFA 两算子的接口契约

`QuantFlashAttn`（主算子，AI Core）+ `QuantFlashAttnMetadata`（伴生算子，**AICPU**）。
接入流程见 `../vendor-ops-transformer-op.md`，门禁见技能 `vendor-ascend-op`。
本页只放**会被反复查的表与语义**。

## 官方文档在哪

cann/ops-transformer 的算子文档能直接 `curl`，不用开浏览器：

```bash
curl -sfL https://raw.gitcode.com/cann/ops-transformer/raw/master/attention/quant_flash_attn/docs/torchapi_quant_flash_attn.md
```

⚠️ `https://gitcode.com/cann/ops-transformer/raw/master/...`（**没有 `raw.` 前缀**）
返回的是 5KB HTML 壳，不是正文。认准 `raw.gitcode.com`。换算子只改路径里的
`attention/<op_name>/docs/torchapi_<op_name>.md`。

## descale 的 shape 匹配关系（quant_mode=1 / MXFP8）

这张表是接 QFA 最容易配错的地方，官方文档里单独列了：

| 参数 | layout | shape | 场景 |
| --- | --- | --- | --- |
| `q_descale` | TND | `(Q_T, Q_N, ⌈D/64⌉, 2)` | Prefill，推荐 `G*Q_S > 80` |
| `q_descale` | N2TGD | `(KV_N, Q_T, G, ⌈D/64⌉, 2)` | Decode，推荐 `G*Q_S <= 80` |
| `k_descale` / `v_descale` | 见官方文档 | — | PA 路径下 K/V 已是 FP8，读侧不再量化 |

其中 `G = Q_N / KV_N`。文档另有 `mask_mode` / `quant_mode` 枚举、两个接口的完整原型、
TND+PA 的调用示例。

**`max_seqlen_q` 是「单条序列」的 q 长度上限，不是整批 token 数。** 文档明确算子
不校验 `seqused_q` 的最大值与它一致，要求使用者自行保证
`max(seqused_q) <= max_seqlen_q`。（AICPU 实现里是 `std::max(attr, 逐 batch 长度)`，
传小会自愈、传大不会 walk back——所以它是**语义问题不是性能问题**。）

## tilingKey 六维盘点

`GenTilingKey()`（`quant_flash_attn_tiling_mxfp8.cpp`）拿六维拼 key。逐维确认 Python
侧能不能动：

| 维 | 由什么决定 | 状态 |
| --- | --- | --- |
| `inputLayout` | `layout_q` | TND，定死正确 |
| `config` | `AdjustSinnerAndSouter` | D=256 → sOuter 64 / sInner 256，定死正确 |
| `quantMode` | `decodeS1GMerge_ = (layoutQDescale == N2TGD)` | ✅ 已按 `G*Q_S <= 80` 切 |
| `hasAttenMask` | `maskMode != NO_MASK` | ✅ 已在 `max_query_len == 1` 时传 0 |
| `kvLayoutType` | `layout_kv` | PA_NZ，定死正确 |
| `isFd` | `flashDecodeFlag_` | ❌ `SplitPolicy` 里写死 false，算子侧 |

⚠️ **`AdjustSinnerAndSouter` 是个幌子**：签名收 `maxSeqQ` / `maxSeqKv` / `maskMode` /
`win`，但 `vHeadDim == 256` 的第一个分支就 `return`，四个入参全不看
（`op_host/qfa_adjust_sinner_souter.h:62-65`，`SOUTER_64 = 64` / `SINNER_256 = 256`）。
所以 `max_seqlen_kv=-1` 不影响 tiling，别去补它。

⚠️ **`scripts/27B.sh` 里关于这点的注释是错的**（它说 `MAX_MODEL_LEN` 之所以要单变量实验，
是因为"captured QuantFlashAttn bakes that constant in -- `AdjustSinnerAndSouter` tiles on it"）。
按源码，D=256 路径完全不吃 `maxSeqKv`。`MAX_MODEL_LEN` 确实要单变量调，但理由是它在
**metadata plan** 与 `_qfa_max_seqlen_kv` 里起作用，不是主算子的 host tiling。

### 表中 ✅ 那几维是什么时候动的

2026-09-16 三个 commit 落在 `junlin-c8-mxfp-16278`：

| commit | 改的哪一维 |
| --- | --- |
| `c7c2cd791` | plan 复用（一步一次、跨层共享） |
| `9c6a61b6b` | `hasAttenMask`：`max_query_len == 1` 时传 0 |
| `b5e7ee06b` | V scale 挪到 cache 初始化——见 [`../cases/mtp-accept-graph.md`](../cases/mtp-accept-graph.md) |

⚠️ **全部只验了调用契约，真机 A/B 未做。** 引用这三项收益前先补上真机对照。

### 剩下的优化都在算子侧

`out=` 参数（省掉输出拷贝）、strided `q_descale`（省掉 N2TGD 的 `permute+contiguous`）、
PA_NZ + MXFP8 的融合 scale scatter，以及量级最大的 `flashDecodeFlag_`（见技能
`vendor-ascend-op` 门 6：host tiling / kernel `if constexpr` / AICPU 三处配套关闭）。

`ScatterPaKvCacheWithKScale` 帮不上：只支持 BNBD，且 `key_scale` 是 per-token-head 的
单个 FLOAT，不是 MXFP8 的块缩放。

## metadata plan 的共享边界

**plan 只依赖**：头数 / 头维 / `quantMode` / `cu_seqlens_q` / `seqused_kv` / `maskMode` /
`win` / `layout` / `deviceInfo`——**没有逐层量**。主算子侧 `this->Input("metadata")` 是
只读输入；`v_descale` 只在 `layoutKv == "TND"` 做校验，PA_NZ 跳过。

⇒ **一步只调一次、全部注意力层共用**，捕获期同理（调用录在捕获区内，流序保证写先于读）。
已经照这条砍掉的那批小算子、以及为什么不能把剩下的挪到 `prepare_inputs`，见
`../cases/qfa-pre-ops.md`。

## 源码交叉验证

vendor 进来的一份 csrc 在 `vllm-ascend/junlin-qfa/csrc/attention/quant_flash_attn/`
（只有 `junlin-qfa` 有，`junlin-c8-mxfp` 系走外部 `cann_ops_transformer` 包）。

- host checker：`op_host/checkers/quant_checker.cpp` 里的 `expected` 数组是上面 descale
  表的**代码版**，核对文档时很好用。
- tiling：`op_host/arch35/quant_flash_attn_tiling_mxfp8.cpp`。
- ⛔ **不读也不改算子本体**（`op_api/` `op_host/` `op_kernel/` `common/`）；
  自己的适配层是 `quant_flash_attn_torch_adpt.h`。

换 CANN 包 / rebase vendor 之后**先跑契约体检**再谈别的：

```bash
python scripts/checks/qfa_op_contract.py
```
