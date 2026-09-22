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

**Python wrapper**（`cann_ops_transformer/ops/attention/quant_flash_attn/quant_flash_attn.py`）：

```python
max_schedule_size = _calculate_max_schedule_size(batch_size, num_heads_kv)
output = torch.empty((2, max_schedule_size), dtype=torch.int32, device="npu")
```

`_calculate_max_schedule_size` 的 docstring 明确写了它的假设：

> dim0 按 sectionNum 最坏值(**batch\*num_heads_kv**)动态计算

**AICPU kernel**（`quant_flash_attn_metadata_aicpu.cpp`）：

```cpp
bool isDecode = (layoutQDescale_ == "N2TGD");
baseInfo.kvHeadNum = isDecode ? numHeadsKv_ : numHeadsQ_;   // ← TND 取 numHeadsQ
```

而 `CalcGridInfoSection`（`load_balance/section_stream_k/section_stream_k_impl.h`）
的内层循环正是按 `baseInfo.GetKvHeadNum()` 计数：

```cpp
for (uint32_t bIdx = 0; bIdx < baseInfo.GetBatchSize(); bIdx++) {
    for (uint32_t n2Idx = 0; n2Idx < baseInfo.GetKvHeadNum(); ++n2Idx) {
        if (超出 l2Byte) { gridInfo.sectionNum++; }
    }
}
```

于是 `layout_q_descale="TND"`（prefill）下 sectionNum 的上界是
`batch * num_heads_q`，而 buffer 按 `batch * num_heads_kv` 分配。**GQA 下两者差
G = num_heads_q / num_heads_kv 倍**，`FaMetaData::Clear()` 直接写出界。

decode（`N2TGD`）两边都用 `num_heads_kv`，所以不复现。

## 为什么旧版不暴露

section 切分被 `param.l2Byte` 门控：

```cpp
if (m_param.l2Byte == 0U) { gridInfo.sectionNum = 1; return; }   // 旧版恒走这里
```

旧版 `param.l2Byte = 0`、`fdOn = 0`，**sectionNum 恒为 1**，需求是一个与头数无关的
常量；旧 wrapper 又是固定分配 `(4096,)` 一维 int32，怎么都够。新版 MXFP8 下
`l2Byte = 96MB`、`fdOn = true`，切分真正生效，缺口才暴露出来。

实测印证（见下）：旧包 metadata 恒为 `(4096,)`，新包为 `(2, 8192)`。

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

`_calculate_max_schedule_size` 按 layout 选头数，与 AICPU 侧的
`baseInfo.kvHeadNum = isDecode ? numHeadsKv : numHeadsQ` 对齐：

```python
heads = num_heads_kv if layout_q_descale == "N2TGD" else num_heads_q
max_schedule_size = _calculate_max_schedule_size(batch_size, heads)
```

另建议让 AICPU 侧那段容量自检不要被静默跳过。目前是：

```cpp
if (rowSize <= 0 || dimNum <= 0) { /* 回退读 TensorShape，注释自承"部分平台不填充" */ }
if (dimNum > 0 && rowSize > 0) { /* 容量校验 */ }   // ← shape 拿不到就整段跳过
```

shape 取不到时校验被整体跳过，越界就从一条清晰的 `KERNEL_LOG_ERROR` 变成 AICPU
abort，再经由 device 挂掉伪装成下游算子的错误，定位成本很高。
