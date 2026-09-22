# QFA metadata 在 prefill 下越界写，AICPU abort

提给 cann/ops-transformer。已在 A5 真机上复现，附最小复现脚本与两份对照输出。

## 现象

PD 分离部署（P: TP8，D: TP4+DP2，Qwen3.5-397B），服务拉起正常，**一发请求 P 节点就挂**：

```
AI CPU kernel execution failed, device_id=5, stream_id=61, task_pos=352,
soName=libcpu_kernels.so, funcName=RunCpuKernel, kernelName=QuantFlashAttnMetadata,
errorCode=0x2a, argsSize=2145, aicpuKernelType=TS_AICPU_KERNEL_AICPU(2)
```

AICPU abort 打死 device 之后，同一条 stream 上后续算子全部 507018
（`aclnnChunkGatedDeltaRuleFwd` / `aclnnSwiGlu` / `aclnnCumsum`），Python 栈因此落在
GDN、SwiGlu 这些**后面才跑的算子**上，看不出和 QFA 有关。三个 rank 的 TraceBack 里
AICPU kernel 名都是 `QuantFlashAttnMetadata`，这是唯一的第一现场。

旧版 ops-transformer 不复现，换到新版后必现。

## 根因

分配方与写入方对"用哪个头数"的理解不一致。

**Python wrapper**（安装后位于 `$CANN_HOME/python/site-packages/cann_ops_transformer/`，模块 `cann_ops_transformer.ops.attention.quant_flash_attn.quant_flash_attn`）：

```python
max_schedule_size = _calculate_max_schedule_size(batch_size, num_heads_kv)
output = torch.empty((2, max_schedule_size), dtype=torch.int32, device="npu")
```

`_calculate_max_schedule_size` 的 docstring 明确写了它的假设：

> dim0 按 sectionNum 最坏值(**batch\*num_heads_kv**)动态计算

**AICPU kernel**（`attention/quant_flash_attn_metadata/op_kernel_aicpu/quant_flash_attn_metadata_aicpu.cpp:200`）：

```cpp
bool isDecode = (layoutQDescale_ == "N2TGD");
baseInfo.kvHeadNum = isDecode ? numHeadsKv_ : numHeadsQ_;   // ← TND 取 numHeadsQ
```

而 `CalcGridInfoSection`
（`attention/common/op_kernel/load_balance/section_stream_k/section_stream_k_impl.h:333`）
的内层循环正是按 `baseInfo.GetKvHeadNum()` 计数：

```cpp
for (uint32_t bIdx = 0; bIdx < baseInfo.GetBatchSize(); bIdx++) {
    ...
    for (uint32_t n2Idx = 0; n2Idx < baseInfo.GetKvHeadNum(); ++n2Idx) {   // L342
        if (tokenSize != 0 && !IsWithinTolerance(tokenLimit, INT64_ZERO, tokenSize + singleHeadCost)) {
            gridInfo.sectionBnIdx.emplace_back(ToOutputLayoutBnIdx(bn2Idx, baseInfo));
            gridInfo.sectionNum++;
        }
        ...
    }
}
gridInfo.sectionBnIdx.emplace_back(baseInfo.GetBatchSize() * GetHeadNum(baseInfo));   // L353
gridInfo.sectionNum++;
```

（L353 的 `GetHeadNum(baseInfo)` 在 `m_param.outputLayout != BN1_S1` 时就是
`baseInfo.GetKvHeadNum()`，见 `section_stream_k_impl.h:503`；AICPU 设的正是
`param.outputLayout = OutputLayout::BN2_S1G`，所以两者同值。）

于是 `layout_q_descale="TND"`（prefill）下 sectionNum 的上界是
`batch * num_heads_q`，而 buffer 按 `batch * num_heads_kv` 分配，**GQA 下差
G = num_heads_q / num_heads_kv 倍**。

写入没有任何边界检查（`quant_flash_attn_metadata.h:63`）：

```cpp
FaMetaData(uint32_t aicNum, uint32_t aivNum, uint32_t sectionNum, void *metadataPtr)
    : headMedata(static_cast<FA_METADATA_T *>(metadataPtr)),
      faMetadata(headMedata + METADATA_STRIDE),
      fdMetadata(faMetadata + sectionNum * aicNum * METADATA_STRIDE) {}

void Clear() {
    for (size_t i = 0; i < METADATA_STRIDE; ++i)                      headMedata[i] = 0U;
    for (size_t i = 0; i < sectionNum * aicNum * METADATA_STRIDE; ++i) faMetadata[i] = 0U;
    for (size_t i = 0; i < sectionNum * aivNum * METADATA_STRIDE; ++i) fdMetadata[i] = 0U;
}
```

写入总量正是 `METADATA_STRIDE + sectionNum * (aicNum + aivNum) * METADATA_STRIDE`，
即 AICPU 自己那句容量校验里的 `needSize`。sectionNum 一旦超出 buffer 能容纳的
section 数，`Clear()` 当场写出界。

decode（`N2TGD`）两边都用 `num_heads_kv`，所以不复现。

## 为什么旧版不暴露

section 切分被 `param.l2Byte` 门控：

```cpp
if (m_param.l2Byte == 0U) { gridInfo.sectionNum = 1; return; }   // 旧版恒走这里
```

旧版 `param.l2Byte = 0`、`fdOn = 0`（新版见
`quant_flash_attn_metadata_aicpu.cpp:227-234`，`quant_mode==1` 才走
`l2Byte = 96MB / fdOn = true`），**sectionNum 恒为 1**，需求是一个与头数无关的常量
`16 + (aic + aiv) * 16`。实测旧包返回的 metadata 在 512…32768 六个长度下恒为
一维 `(4096,)`，足够覆盖；新包为二维 `(2, 8192)`，由
`_calculate_max_schedule_size` 按 batch 算出。切分一旦真正生效，缺口就暴露。

## 复现

`scripts/checks/qfa_metadata_capacity.py`，只调 metadata 算子，不需要 q/k/v、KV cache
或模型。默认参数是 397B per-rank @TP8：`num_heads_q=4 num_heads_kv=1 head_dim=256`，
即 G=4。

```bash
python3 scripts/checks/qfa_metadata_capacity.py
```

**旧包（GREEN，exit 0）**：六个长度 512…32768 全过，metadata 恒定不变。

```
== SCAN (layout_q_descale=TND, one subprocess per length) ==
  seq_len=512     ok   OK shape=(4096,) numel=4096 dtype=torch.int32
  ...
  seq_len=32768   ok   OK shape=(4096,) numel=4096 dtype=torch.int32
```

**新包（RED，exit 1）**：

```
  formula inputs from the installed wrapper: stride=16 aic=32 aiv=64
  _calculate_max_schedule_size(4, 1) = 8192
    needed, decode  (sectionNum<=4*1=4)  : 6160   fits
    needed, prefill (sectionNum<=4*4=16) : 24592  SHORT

  seq_len=16384   ok   OK shape=(2, 8192) numel=16384
  seq_len=32768   DIED exit=1 [Error]: The aicpu execution is abnormal.

== CONTROL (same batch=4 seq_len=32768, only layout_q_descale differs) ==
  N2TGD (decode template) ok   OK shape=(2, 8192) numel=16384
```

CONTROL 是单变量对照：**同一个 batch、同一个 seq_len，只改 `layout_q_descale`**，
TND 死、N2TGD 活。排除了序列长度、batch 和一般意义上的容量不足。

越界点可以精确预测，与实测逐项吻合：

```
sectionNum = ceil(batch * num_heads_q * 4 * head_dim * S / l2Byte)
越界条件   = 16 + sectionNum * (aic + aiv) * 16 > max_schedule_size
```

| seq_len | sectionNum | need | alloc | 实测 |
| --- | --- | --- | --- | --- |
| 8192 | 2 | 3088 | 8192 | ok |
| 16384 | 3 | 4624 | 8192 | ok |
| **32768** | **6** | **9232** | 8192 | **DIED** |

预测临界 `S > 30721`，实测第一个越过它的 32768 即崩。

## 建议修法

要改的是 wrapper 的分配侧，文件
`attention/quant_flash_attn/torch_extension/quant_flash_attn.py`（行号对
`f7fe4ec0d`）。让 `_calculate_max_schedule_size` 按 layout 选头数，与 AICPU 侧的
`baseInfo.kvHeadNum = isDecode ? numHeadsKv : numHeadsQ` 对齐。

**两处调用都要改，不能只改一处**：

| 行 | 位置 | 作用 |
| --- | --- | --- |
| L242-244 | `quant_flash_attn_metadata_meta`（`@torch.library.register_fake`） | torch.compile 下 fake tensor 的 shape |
| L394-395 | `quant_flash_attn_metadata` 本体 | 真机上实际 `torch.empty` 出来的 buffer |

只改 L394，图模式下 fake tensor 与真实 tensor 的 shape 会对不上；只改 L242，真机
照样越界。

```python
def _calculate_max_schedule_size(batch_size, num_heads_kv, num_heads_q=None, layout_q_descale=None):
    ...
    # AICPU 侧 baseInfo.kvHeadNum = isDecode ? numHeadsKv : numHeadsQ，
    # CalcGridInfoSection 的内层循环按它计数，所以 sectionNum 的上界随 layout 变。
    heads = num_heads_kv
    if num_heads_q is not None and layout_q_descale != "N2TGD":
        heads = num_heads_q
    fa_size = aic_num * METADATA_STRIDE * batch_size * heads
    fd_size = aiv_num * METADATA_STRIDE * batch_size * heads
```

判据写成 `!= "N2TGD"` 而不是 `== "TND"`，因为 **L242 的 meta kernel 里
`layout_q_descale` 可能是 `None`** —— 它没有做真实分配侧 L387 那样的
`layout_q_descale = "BSND" if layout_q_descale is None else layout_q_descale`
归一化。`None != "N2TGD"` 取 `num_heads_q`，偏大偏安全，两侧结果也一致。

### 附带：`_get_core_nums()` 的无 NPU 默认值与真机不符

```python
def _get_core_nums():
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        return 36, 72          # ← A5 真机是 32, 64
    props = npu.get_device_properties()
    return props.cube_core_num, props.vector_core_num
```

A5 上 `get_device_properties()` 返回 **32 / 64**，而无 NPU 分支的默认值是 36 / 72。
两者对齐后常常相同（batch=4、num_heads_kv=1 时都落到 8192），但不总是——batch=10
时一个 20480、一个 16384。meta kernel 与真实分配若分处有无 NPU 的两种环境，shape
就会对不上。建议把默认值与真机对齐，或在拿不到设备属性时按上界取值。

（算子自带的 C++ 示例 `test_aclnn_quant_flash_attn_metadata.cpp` 里也写死
`(36 + 72)`，同样偏离真机值。）

另建议让 AICPU 侧那段容量自检不要被静默跳过。目前是：

```cpp
if (rowSize <= 0 || dimNum <= 0) { /* 回退读 TensorShape，注释自承"部分平台不填充" */ }
if (dimNum > 0 && rowSize > 0) { /* 容量校验 */ }   // ← shape 拿不到就整段跳过
```

shape 取不到时校验被整体跳过，越界就从一条清晰的 `KERNEL_LOG_ERROR` 变成 AICPU
abort，再经由 device 挂掉伪装成下游算子的错误，定位成本很高。
