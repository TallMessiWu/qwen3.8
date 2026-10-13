# C8 / QFA 主线配置与缓存格式

> **当前基线（2026-10-13）**：官方 `upstream/main`，本机 worktree
> `vllm-ascend/upstream-main` @ `fbb4d867d22095047473e3c7ca9db053f5f4c7b0`。
> [PR #17180](https://github.com/vllm-project/vllm-ascend/pull/17180) 已于北京时间
> 2026-10-07 合入，merge commit 为 `a7f26e5c4e4d9dd7f20fa114f351860bd6552731`。
> 这里是源码核对结果，本轮没有跑 NPU 推理、PD 传输或性能测试。

## 配置入口

| 路径 | 当前主线入口 | KV cache | 注意力算子 |
| --- | --- | --- | --- |
| C8-MXFP | `--kv-cache-dtype mxfp8` | FP8 E4M3 + E8M0 scales，PA_NZ | QFA metadata + QFA 主算子 |
| 普通 KV 基线 | 不启用 `mxfp8`，如 `--kv-cache-dtype auto`，并核对权重是否还有其他 KV recipe | 按模型与 cache 配置确定 | 普通 Ascend attention 路径 |

启用条件来自 `utils.is_c8_mxfp_kv_quant` 和 `ModelSlimConfig.enable_mxfp_c8_quant`。
权重里的 `K_DYNAMIC_V_STATIC_MXFP8_PER_CHANNEL` / `QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL`
是 recipe 描述，**不是主线的启用开关**；缺少 recipe 时源码会使用 neutral V scale，
该路径的精度仍需真机验证。启用时由 `AscendC8MXFPKVCacheAttentionMethod.create_weights`
安装 C8 backend，不再用旧脚本的 `QFA=1` / `VLLM_ASCEND_DISABLE_C8_MXFP` 推断主线行为。

本仓 `27B.sh` 等旧脚本仍保留 `C8=0` 的已删除开关检查，直接跑它可能报错；本次没有修改
这些启动脚本。过去「C8=0 基线拿不到」是旧脚本状态，不代表当前主线无法选择普通 KV。
对照实验要固定 prefix caching、权重、并行配置和 CANN 版本；旧文要求关闭 prefix caching
是当时的实验约束，本轮没有验证新主线的全部 prefix-cache 组合。

## cache 的物理组织与张量格式

**一个 512-token kernel block 对应一个 packet**：

```text
[K | K_scale | V | V_scale | padding]
```

`mxfp_paged_cache_views` 在 raw buffer 上用 `as_strided` 一次建立四个视图；它们共享
storage，首维 stride 是整包 pitch。padding 是否存在由 cache spec 的物理页大小决定。
这里的 Bn 是 kernel block 数，Bs 是 kernel block size（512），N 是本 rank 的 KV 头数。

| 视图 | 类型 | shape |
| --- | --- | --- |
| K | FP8 E4M3 | `[Bn, N, Dk/32, Bs, 32]` |
| V | FP8 E4M3 | `[Bn, N, Dv/32, Bs, 32]` |
| K scale | uint8 保存 E8M0 字节 | `[Bn, N, Bs/16, Dk/64, 16, 2]` |
| V scale | uint8 保存 E8M0 字节 | `[Bn, N, Dv/16, Bs/64, 16, 2]` |

scatter 接收本步量化后的 `[T, N, D]` K/V，设置 `cache_mode="PA_NZ"`，直接写上述
五维 strided 视图。输入/输出的同字节大小 `view(int8)` 只是 bitcast，不是再次量化。
QFA 直接读取这些视图；不能套用旧分支的连续四维 cache 或整库 transpose 逻辑。
PA_NZ 是显式的数据排列，不能仅凭名字等同于 `npu_format_cast` 的隐藏 FRACTAL_NZ 格式。

K scale 随新 token 写入。V scale 是 checkpoint 的静态 per-channel scale：初始化时
填充；主线 #17956 还会在 hybrid 页从 Mamba 重新分配给 attention 时恢复被覆盖的 scale。
这不是每个 decode 步重排全部历史 cache。

## P/D 的格式与转换频率

**P 和 D 使用同一主线实现、匹配 cache 配置时，KV cache 都是 PA_NZ。**
PrefillNoCache 也先写页，再调用 QFA 读分页 cache，没有另一个 TND 历史 KV 存储格式。

| 环节 | 数据处理 | 频率 |
| --- | --- | --- |
| P prefill / D decode 的 cache 写入 | 本步新 K/V：TND → PA_NZ scatter；写 K scale | 每个 cache owner attention 层、每步有新 KV 时 |
| P → D 传输 | Mooncake 按各视图的地址、block 长度和物理 stride 搬原始块，含两份 scale | 请求需要加载远端 prompt KV 时 |
| D QFA 读取 | 直接使用已分配的 PA_NZ 视图 | 每层每步，没有历史 KV 的全量格式转换 |
| Q scale | TND 或按 `G * max_query_len <= 80` 重排为 N2TGD | 本步、每层；不是收到的 KV |
| V scale | 初始化填充，hybrid 页复用时恢复 | cache 初始化 / 页重新分配 |

Q 和输出的 layout 在 P/D 都是 TND。通常长 prefill 的 Q scale 是 TND，短 decode 的
Q scale 是 N2TGD；条件按长度和 GQA group size 选择，不能只按 P/D 角色判断。

当前 C8 backend 只接受 Mooncake connector 路线，并不证明所有 PD 配置均已验收。
不同 TP 的通用接收重排仍有 K/V 二元组假设，四视图 C8 路径应单独检查。connector 的
接收后重排在一次请求的 pulls 完成后触发，不是每个 decode 步重复处理历史 KV。
另一个 `enable_kv_nz` 开关受 MLA / D 节点条件限制，不是本页 C8-QFA 的 P/D 格式开关。

## 源码入口与验证边界

以下均相对 `vllm-ascend/upstream-main/`：

| 文件 / 函数 | 用途 |
| --- | --- |
| `vllm_ascend/attention/attention_c8_mxfp.py` | QFA backend、packet 布局、scatter、Q scale 选择和两个算子的调用 |
| `vllm_ascend/quantization/methods/kv_cache/mxfp_c8.py` | backend 安装、checkpoint V scale 加载 |
| `vllm_ascend/quantization/configs/modelslim_config.py` | `mxfp8` 显式启用与 recipe 映射 |
| `vllm_ascend/worker/model_runner_v1.py` | cache spec、共享内存视图、V scale 初始化与页复用恢复 |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py` | cache 注册、按 block stride 传输、接收后分片重排 |

QFA 两算子从 `cann_ops_transformer.ops` 导入；当前主线 `csrc/` 没有 vendor 的 QFA
源码。`junlin-qfa/csrc/attention/quant_flash_attn/` 只是一份历史实现，不能替代真机所装
CANN 包的契约。接口表见 [`qfa-op-contract.md`](qfa-op-contract.md)。

旧的 `junlin-c8-mxfp` / `junlin-qfa` 是 PA_BBND；`16278/16614` 系虽是 PA_NZ，
但 cache 分配组织不同。旧分支的性能、单测数量和 hybrid 容量预期值仍只对其原版本有效，
历史记录见 [`qfa-integration.md`](../cases/qfa-integration.md) 和
[`hybrid-c8-kv-capacity.md`](../cases/hybrid-c8-kv-capacity.md)。
