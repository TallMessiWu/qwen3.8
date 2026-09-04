# 从 ops-transformer 接入自定义算子到 vllm-ascend

以 `quant_flash_attn`（QFA）为样板，记录一个 ops-transformer 算子进入 `vllm-ascend/csrc`
并被 Python 侧调用起来的完整路径。最后一节是给 **FIA（`fused_infer_attention_score`）**
接入的差异清单——FIA 比 QFA 难，别照抄步骤就开工。

参考实现：分支 `junlin-qfa`（`vllm-ascend/junlin-qfa` worktree），三个提交，
相对 `upstream/main` 共 154 个文件。其中 **51000+ 行是原样拷贝的算子源码**，
真正需要动脑的胶水代码不到 800 行。

---

## 0. 先搞清楚：接的是什么

QFA 接入实际是两个算子：

| 算子 | 类型 | 作用 |
| --- | --- | --- |
| `QuantFlashAttn` | AI Core | 主算子，做 attention |
| `QuantFlashAttnMetadata` | **AICPU** | 先跑，算出切分计划（split plan）写进一个 int32 buffer，主算子读它 |

metadata 算子必须先于主算子调用，且**两次调用的参数集必须完全一致**——kernel 侧不做跨调用校验，
不一致不会报错，会算错或者踩出 AI Core invalid GM address。

接入分四层，缺一层就跑不起来：

```
① 算子源码 vendor      csrc/attention/<op>/          （拷贝 + 改 include 路径）
② 构建接线            build_aclnn.sh / CMakeLists   （让 csrc 的构建系统认识它）
③ torch 绑定          torch_binding{,_meta}.cpp     （手写，让 torch.ops 能调）
④ Python 接入         attention_v1.py / envs.py     （在 vLLM 里真正用上）
```

---

## ① 算子源码 vendor

### 拷什么

从 ops-transformer 拷 `attention/quant_flash_attn/` 与 `attention/quant_flash_attn_metadata/`
到 `csrc/attention/` 下，**去掉 `tests/` 和 `torch_extension/`**。

`torch_extension/` 不用是有意的：它是 ops-transformer 自己那套 torch 扩展，
与 vllm-ascend 的 `torch_binding.cpp` 机制不兼容。绑定要自己写（见 ③），
但它的 Python 实现是**参数语义的权威参考**，读它，别用它。

### 关键决策：common 做成算子私有快照

算子源码里大量 `#include "../../../common/op_kernel/..."` 指向 ops-transformer 的
`attention/common/`。csrc 里虽然也有 `csrc/attention/common/`，但那是**旧切片，只有 23 个文件**，
而 ops-transformer 的有 113 个，且部分同名文件内容已经分叉。直接复用会编不过或行为不一致。

做法是把用到的 common 文件拷成**算子私有的一份快照**：

```
csrc/attention/quant_flash_attn/common/     ← 59 个文件，来自 ops-transformer/attention/common/
csrc/attention/quant_flash_attn_metadata/common/
```

代价是同一份 common 在仓里存了两份；收益是这个算子的依赖被完全冻结，
升级它不会波及别的算子，别的算子改 common 也影响不到它。**推荐延续这个做法。**

对应地，`op_host/CMakeLists.txt` 里要**删掉**上游的共享依赖声明：

```cmake
# 删掉这行（它会去拉 csrc/attention/common）
set(quant_flash_attn_depends attention/common CACHE INTERNAL "...")
```

### 源码里唯一要改的东西：include 路径

拷进来之后，算子源码**只改 include 路径，不改任何逻辑**。因为 common 从
`attention/common/`（算子的兄弟目录）变成了 `<op>/common/`（算子的子目录），
相对路径少一层：

```c
-#include "../../../common/op_kernel/vector_common.h"
+#include "../../common/op_kernel/vector_common.h"
```

同时把上游为了兼容两种目录布局写的 `#if __has_include(...)` 双分支**拉平成单分支**——
快照布局是确定的，留着双分支只会在下次升级时误导人：

```c
-#if __has_include("../../../common/op_kernel/arch35/xxx.h")
-#include "../../../common/op_kernel/arch35/xxx.h"
-#else
 #include "../../common/op_kernel/arch35/xxx.h"
-#endif
```

改动范围（QFA 实测）：`op_host/` 5 个文件、`op_kernel/` 10 个文件、metadata 1 个文件，
全部只是 include 行。

### 两个特例

**`util.h` 撞 CANN 内置头。** 上游写的是裸 `#include "util.h"`，在 csrc 的 include 路径下
会命中 CANN 内置 FA 的同名 `util.h`，那份的 `LayOutTypeEnum` 是旧的，缺 `LAYOUT_NTD` /
`LAYOUT_NTD_TND`，症状是编译期报枚举值不存在。

解法：从 ops-transformer 的 **`common/include/op_kernel/util.h`**（注意是仓根的 `common/`，
不是 `attention/common/`）拷到 `<op>/common/include/op_kernel/util.h`，改成显式相对路径：

```c
-#include "util.h"
+// Embedded copy: the bare "util.h" collides with CANN's built-in FA util.h,
+// whose older LayOutTypeEnum lacks LAYOUT_NTD/LAYOUT_NTD_TND.
+#include "../../include/op_kernel/util.h"
```

**tiling 基础设施进 csrc 公共目录。** 这两个文件从 ops-transformer 的
`common/include/op_host/` **原样拷贝**到 `csrc/common/include/op_host/`：

```
tiling_base.h
tiling_templates_registry.h
```

它们是 tiling 模板注册机制的底座，不适合做成算子私有（多个算子都要用同一套注册表）。

---

## ② 构建接线

### 算子名进构建清单

`csrc/build_aclnn.sh`，ascend950 分支的 `CUSTOM_OPS_ARRAY`：

```bash
     "sparse_attention_score"
     "mla_prolog_v3"
+    "quant_flash_attn"
+    "quant_flash_attn_metadata"
```

`csrc/CMakeLists.txt` 不用改——`add_subdirectory(attention)` 会自动发现
`csrc/attention/CMakeLists.txt` 里 glob 出来的子目录，算子自带的 `CMakeLists.txt` 就够了。

### SOC 版本映射

`csrc/cmake/scripts/util/opdesc_parser.py` 的 `SOC_TO_SHORT_SOC_MAP` 要认识实际的芯片型号串，
否则 opdesc 解析阶段找不到 short soc：

```python
     "ascend950": "ascend950",
+    "ascend950pr_9589": "ascend950",
+    "ascend950pr_9599": "ascend950",
+    "ascend950dt_9582": "ascend950",
```

### 坑一：`Ops::Base::ToString` 未定义 → 全包算子丢 tiling

**这是最贵的一个坑，症状离病因极远。**

ops-transformer 的 checker 代码（`base_checker.cpp` / `quant_checker.cpp`）调用
`Ops::Base::ToString`，上游是从 CANN 的 **opbase 库**链进来的。**csrc 不链 opbase。**

后果链条：符号未定义 → `libcust_opmaster_rt2.0.so` 带着一个 undefined symbol →
TBE dlopen 这个 tiling so 失败 → **整个包里所有走标准机制的算子全部丢失 tiling 注册**，
刷屏 `do not registe tiling struct`。看起来像是把仓搞崩了，实际只是一个符号。

解法是本地补一份实现（`csrc/attention/quant_flash_attn/op_host/qfa_ops_base_compat.cpp`），
语义照抄 ops-transformer `fused_infer_attention_score_tiling_utils.h` 里的 inline 版本，
并在 `op_host/CMakeLists.txt` 里挂进 tiling 目标：

```cmake
target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
    ${ARCH35_CHECKER_SRC_FILES}
    qfa_tiling_info_parser.cpp
    qfa_ops_base_compat.cpp   # ← 补 Ops::Base::ToString
)
```

排查工具已经写好：`scripts/setup/diag_qfa_tiling_registry.sh`，
对构建产物跑 `ldd -r` 一次列出所有未解析符号，别一个一个 dlopen 试。

### 坑二：AICPU 的 CMake 签名不同

csrc 的 `func.cmake` 提供的是老签名 `add_aicpu_cust_kernel_modules(op_name sources jsons)`，
ops-transformer 用的是新的单参数形式 + `target_sources`。metadata 算子的
`op_kernel_aicpu/CMakeLists.txt` 要改写：

```cmake
-  set(OBJ_NAME ${OP_NAME}_cust_obj)
-  add_aicpu_cust_kernel_modules(${OBJ_NAME})
-  target_sources(${OBJ_NAME} PRIVATE ${AICPU_SRC})
+  add_aicpu_cust_kernel_modules(quant_flash_attn_metadata ${AICPU_SRC} ${JSON_FILE})
```

### 坑三：typos 钩子拦拼写

算子源码里的拼写错误（`colums` / `pading` / `schduler` / `fuction`）会被 pre-commit 的
typos 钩子拦下。**不要改算子源码**——改了就再也 diff 不出与上游的差异了。
加白名单 `.pre-commit-config.yaml`：

```yaml
-'-L', 'CANN,cann,...,LoadIn'
+'-L', 'CANN,cann,...,LoadIn,colums,pading,schduler,fuction'
```

---

## ③ torch 绑定（手写）

三个文件，都在 `csrc/`：

### `attention/<op>/<op>_torch_adpt.h`

算子的 C++ 入口，负责算输出 shape、准备参数、`EXEC_NPU_CMD` 调 aclnn。
参数语义照着 ops-transformer 的 `torch_extension/` 抄。

**坑四：layout 字符串指针悬空。** aclnn executor 会把属性指针一直持有到 ACL graph
capture 结束，所以传进去的 `char*` 必须活过这次调用。`c10::string_view` 指向的内存不保证活着。

解法是进程级驻留（layout 词表很小，就 TND/BSND/PA_BBND 那几个）：

```cpp
inline char *InternedLayout(c10::string_view layout)
{
    static std::mutex mu;
    static std::unordered_set<std::string> pool;
    std::lock_guard<std::mutex> lock(mu);
    const std::string &stored = *pool.emplace(layout).first;
    return const_cast<char *>(stored.c_str());
}
```

同样的坑在 `msa_index_score_torch_adpt.h` 里也有注释记录，是个反复出现的模式。

**坑五：只写分配式重载，图捕获就没有退路。** aclnn 接口本来就是 out 语义——
`aclnnXxxGetWorkspaceSize` 末尾那几个 tensor 参数就是输出，由调用者提供
（QFA 是 `attnOut` / `softmaxLseOptional`，FIA 是 `attentionOut` / `softmaxLse`）。
在 adpt 里用 `at::empty` 替调用者分配当然能跑，但那样 torch 侧就只剩一个「返回新张量」的
重载，而 `torch.npu.graph_task_group` / `graph_task_update`——aclgraph 里 replay 时重发
调用、**连 tiling 一起换**的唯一手段——要求输出是调用者（也就是图）持有的内存。没有 out
重载就用不了它。

QFA 就栽在这：vendor 时只写了分配式重载，后来为了让捕获的算子读到新数据，自己发明了
「捕获期自持 buffer + 每步 `copy_` 刷内容」的替代方案。那套在真机上必崩
（AI Core 野指针，故障地址在图池基址 +179MB~748MB，而所有合法入参都在 +64KB 以内），
排查了很多轮才发现问题在 binding 不在算子——期间一度准备把「QFA 可能不支持 aclgraph」
当结论去问算子团队，方向完全错了。

补 out 变体是 20 行薄封装，不碰 kernel，跳过 shape 推导直接下发：

```cpp
std::tuple<at::Tensor, at::Tensor> npu_xxx_out(
    /* 与分配式重载完全相同的入参 */,
    at::Tensor &attn_out, at::Tensor &softmax_lse)
{
    /* 同样的 TORCH_CHECK 与 InternedLayout */
    EXEC_NPU_CMD(aclnnXxx, ..., attn_out, softmax_lse);
    return std::make_tuple(attn_out, softmax_lse);
}
```

⇒ **vendor 任何可能进图的算子，binding 一开始就出两个重载**：分配式给 eager 用，
out 变体给图捕获用，共用同一行 `EXEC_NPU_CMD`。这是个主动检查项，别等撞了图捕获才发现。
参考：torch_npu 里 49 个 attention 算子只有 4 个有 out 变体，恰好都是要进 aclgraph 的
（`npu_fused_infer_attention_score{,_v2}`、`npu_fusion_attention{,_grad}_v3`）。

### `torch_binding.cpp`

include 头文件 + `ops.def` 写 schema + `ops.impl` 绑到 `torch::kPrivateUse1`。
schema 里可选参数一律 `Tensor?`，带默认值的放 `*,` 之后。

要进图的算子再 def 一个 `.out` overload（见坑五）：out 张量放最后、标 `Tensor(a!)`
表示原地写，返回同样的别名。

```cpp
ops.def(
    "npu_xxx.out(Tensor q, ..., bool return_softmax_lse=False,"
    "            Tensor(a!) attn_out, Tensor(b!) softmax_lse)"
    " -> (Tensor(a!), Tensor(b!))"
);
ops.impl("npu_xxx.out", torch::kPrivateUse1, &vllm_ascend::npu_xxx_out);
```

之后 Python 侧 `torch.ops._C_ascend.npu_xxx.out(...)` 可用，
`torch.ops._C_ascend.npu_xxx.overloads()` 会多出 `'out'`——这是最快的自检。

### `torch_binding_meta.cpp`

meta 实现（只算 shape、不碰数据）+ `ops.impl` 绑到 `Meta`。
**图模式必需**——没有 meta 实现，torch.compile / ACL graph 追踪会直接失败。
逻辑就是 adpt 里 shape 计算部分的 symint 版本，用 `sym_size` / `at::empty_symint`。

出了 out 重载就要一并给它 meta 实现，注册到 `"npu_xxx.out"`。它没有 shape 要推——
输出是调用者给的，原样返回即可。

---

## ④ Python 接入

### `vllm_ascend/envs.py`

新算子一律挂开关，默认关：

```python
"VLLM_ASCEND_ENABLE_QFA": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_QFA", "0"))),
```

### `vllm_ascend/attention/attention_v1.py`

三处：

1. **metadata builder 里 hoist 每步一次的准备工作**（`_attach_qfa_inputs`）。
   一个 step 的所有 full-attention 层喂给算子的参数是同一份，
   每层跑一次 AICPU metadata 算子纯属重复。提到 builder 里跑一次，挂在 `AscendMetadata.qfa` 上。
   这也是主算子能被图捕获的前提——tensor 地址固定下来了。

2. **forward 里加分支**，与 FIA 基线并列，开关控制：

```python
if self.enable_qfa:
    attn_output = self._forward_qfa(...)
else:
    attn_output, _ = DeviceOperator.npu_fused_infer_attention_score(...)
```

3. **图捕获路径**（`full_graph_qfa` + `_update_qfa_graph_params`）。
   **与 FIA 用同一套机制**，别另起炉灶（我们试过，见坑五）：捕获时把 `.out()` 调用包进
   `graph_task_group_begin/end` 拿 handle，把这次调用的张量记进 `graph_params.qfa_params`；
   replay 前 `graph_task_update_begin(update_stream, handle)` → 用**本步的**张量重发整个
   调用 → `graph_task_update_end` → `event.record`。aclnn 每次调用都重算 tiling，所以这样
   连 tiling 一起换掉了，而不只是重绑地址。

   只有那些图内每步自己重算的中间量（QFA 的 `q_fp8` / `q_descale`，由图内量化从本步 query
   产出）原样回传——它们的地址本就该固定。

   **draft 模型走同一条路**。MTP 把 N 个 draft 步并进一张图，捕获顺序是「第 0 步的各层、
   第 1 步的各层……」，replay 时把 `draft_attn_metadatas` 按 `(draft_step, key)` 展平就能
   逐个对上（`_qfa_steps_per_op`）。两处必须注意：graph params 要按 `is_draft_model` /
   `is_draft_model_prefill` 选对应注册表（读 target 的会释放没人 record 的 event，
   表现为死锁）；每个 draft 步有自己的 plan，等 `plan_ready` 要放进循环里。

### `vllm_ascend/compilation/acl_graph.py`

`GraphParams` 加字段存捕获下来的 params / handles / events，**带默认值**，
免得改到已有的位置构造调用：

```python
qfa_params: dict[int, list[Any]] = field(default_factory=dict)
qfa_handles: dict[int, list[torch_npu._C._NPUTaskGroupHandle]] = field(default_factory=dict)
qfa_events: dict[int, list[torch.npu.ExternalEvent]] = field(default_factory=dict)
```

自带一套而不是复用 FIA 的 `attn_params` / `handles` / `events`，是因为那三个的 tuple
布局是 FIA 的。QFA 不需要 `workspaces`。

> **当前状态**：`junlin-qfa-graph` 上 QFA + MXFP8 KV cache + MTP + FULL 图**全开可服务**
> （2026-09-01 实机，Qwen3.5-35B-A3B-mxfp4-c8）。prefill、eager decode、图内 decode
> 全部走 QFA，draft 也走。这部分代码可以当样板读。

### 怎么确认算子真的进了图

`_qfa_serves` 那种「条件判断通过」的日志**证明不了**捕获发生——它在 eager 路径上也会打。
早期有一次就是这样：服务正常、日志好看，而 QFA 从头到尾没进过图，子目标其实没达成。
要一行只有捕获路径才可能打出来的，放在拿到 task handle 之后：

```
[qfa] captured op #7 into task group at tokens=32 draft=False (state=...)
[qfa] update tokens=32 draft=False -> updating {'ops': 10, 'handles': 10, 'events': 10}
```

两条都要有，且 `ops` 等于捕获计数（target = 全注意力层数，draft = 步数 × draft 层数）。
去重键里记得带 `is_draft_model`，否则 draft 的行会被 target 同尺寸的吃掉。

---

## ⑤ 验证闭环

本机没有 NPU，全部在服务器容器里跑。两个脚本已经写好：

**快速迭代（只编两个算子，10–40 分钟）**

```bash
bash scripts/setup/build_qfa_ops.sh /home/hajimi/qwen3.8/vllm-ascend/junlin-qfa
```

失败时它会把 ninja 的 `FAILED:` 块和编译器错误抽出来打印——**不要在滚动输出里找错误**，
并行任务的警告会把根因埋掉。成功判据：`nm -D libcust_opapi.so` 至少 4 个算子符号
（主算子 + metadata，各有 GetWorkspaceSize 和 exec）。

**全量 editable 安装**

```bash
bash scripts/setup/pip_install_qfa.sh /home/hajimi/qwen3.8/vllm-ascend/junlin-qfa
```

**坑五：快速构建的产物会污染全量构建。** 两算子构建留下的 `csrc/build/custom` 是一个
只含这两个算子的 op store，直接复用会让其他所有算子 fall through 到 CANN 内置目录、
参数校验失败（比如 SwigluGroupQuant 3-vs-4 inputs）。全量安装前必须清
`csrc/{build,build_out,output}`——脚本默认就清，`KEEP_BUILD=1` 才复用。
third-party tarball 在 `build/` 之外，清了不会重新下载。

成功判据：`torch.ops._C_ascend.<op>` 存在。之后跑数值验证：

```bash
python scripts/bench/test_qfa_op.py        # 8 个 case 对 golden
python scripts/bench/test_qfa_vs_fia.py        # 与 FIA 对齐
python scripts/bench/test_qfa_vs_fia.py --bench  # 性能对比
```

---

## ⑥ 接 FIA 需要额外知道的

以下都是**已核实的结构事实**，但没有一条在 FIA 上实际编译验证过。

### 头号风险：算子重名

**CANN 内置了 `FusedInferAttentionScore`**，vllm-ascend 现在正是通过
`torch_npu.npu_fused_infer_attention_score` 在调它。vendor 一份同名算子进 custom 包，
OpDef 注册名、aclnn 导出符号（`aclnnFusedInferAttentionScore`）、tiling 注册三处都可能撞。
**开工前先把这件事验证掉**，不要写完 200 多个文件才发现装不上。

仓里有并存先例可循：`csrc/attention/lightning_indexer/`（`OP_ADD(LightningIndexer)`）与
`csrc/attention/lightning_indexer_vllm/`（`OP_ADD(LightningIndexerVllm)`）同包共存。
如果确认会撞，照这个模式重命名——目录名、OpDef 名、aclnn 入口名一起改。

QFA 没遇到这个问题（CANN 里没有 `QuantFlashAttn`），所以样板里没有对应处理。

### 依赖面更大

| | QFA | FIA |
| --- | --- | --- |
| 兄弟目录依赖 | `attention/common` | `attention/common` + `incre_flash_attention` + `prompt_flash_attention` |
| common 依赖文件数 | 59 | QFA 的超集 |
| 芯片代次 | 仅 arch35 | arch22 / arch35 / arch38 |
| 文件数（去 tests/docs/examples） | ~130 | **236** |
| torch_extension | 有（作参考） | **没有** |
| aclnn 入口 | 单一 | **v1 ~ v5 五个版本** |

FIA 的 common 依赖比 QFA 多这几个，做私有快照时别漏：

```
common/op_host/fia_tiling_shape.h
common/op_host/split_core.h        + split_core.cpp     ← 注意有 .cpp
common/op_host/split_core_v2.h     + split_core_v2.cpp  ← 注意有 .cpp
common/op_kernel/CopyInL1.h
common/op_kernel/arch35/flash_attention_score_tiling_regbase_arch35.h
common/op_kernel/arch35/vf/vf_antiquant_w4.h
common/op_kernel/arch35/vf/vf_antiquant_w8.h
common/op_kernel/arch35/vf/vf_post_quant_arch35.h
```

`split_core.cpp` / `split_core_v2.cpp` 是 QFA 没有的情况：FIA 的
`op_host/CMakeLists.txt` 直接在 `target_sources` 里引用了 common 下的 **.cpp 源文件**，
不只是头文件。私有快照必须把它们也拷进来，并同步改 CMakeLists 里的路径。

### 可以裁剪的部分

- **arch22 / arch38 可以不接。** 目标是 ascend950（arch35），另外两代的 op_kernel 和
  tiling 都能砍。但 `op_host/CMakeLists.txt` 里 `target_sources` 那一长串 tiling 源文件
  要**同步裁**——留着会因为找不到文件直接 CMake 失败。arch35 相关文件只有 72 个。
- `attn_infra/` **不是外部依赖**，它在 `op_kernel/arch22/attn_infra/` 下，是 arch22 自带的。
  砍掉 arch22 就一起没了。
- `tests/` / `docs/` / `examples/` 照例不拷（111 个文件）。但 `docs/` 和 `examples/`
  在写 torch 绑定时值得先读——FIA 没有 torch_extension，参数语义只能从这里推。

### 绑定要自己挑版本

FIA 的 `op_api/` 有 `aclnn_fused_infer_attention_score{,_v2,_v3,_v4,_v5}.h` 五个入口。
写 `torch_adpt.h` 前先确定要绑哪一版，看 `op_host/*_def.cpp` 里 OpDef 的输入输出定义
和各版本头文件的参数表对齐——版本之间参数是增量演进的，选错了会在 aclnn 调用时才报错。

### 建议的推进顺序

1. **先验证重名问题**——最小改动 vendor 一个空壳或改名版本，确认能装上、能 `nm -D` 看到符号。
2. 拷源码 + 裁 arch22/arch38 + 做 private common 快照，只求 `build_qfa_ops.sh`
   （改算子名）能编过。此时不写任何 torch 绑定。
3. 编过之后再写 `torch_adpt.h` + 两个 binding，跑全量安装，确认 `torch.ops._C_ascend` 里有它。
4. 最后才动 `attention_v1.py`，挂 `VLLM_ASCEND_ENABLE_*` 开关，与现有 FIA 路径并列。

每一步都有独立的 RED/GREEN 判据，不要跳步——第 ② 步的 tiling 注册坑如果和第 ③ 步的
绑定问题混在一起，日志会难读到没法定位。

---

## 附：改动清单速查

胶水代码（QFA 实测，不含 vendor 的算子源码）：

| 文件 | 改动 |
| --- | --- |
| `csrc/build_aclnn.sh` | +2 行，算子名进 CUSTOM_OPS_ARRAY |
| `csrc/cmake/scripts/util/opdesc_parser.py` | +3 行，SOC 映射 |
| `csrc/common/include/op_host/tiling_base.h` | 新增，原样拷贝 |
| `csrc/common/include/op_host/tiling_templates_registry.h` | 新增，原样拷贝 |
| `csrc/attention/<op>/op_host/CMakeLists.txt` | 删共享 common 依赖，加 opbase 兼容源文件 |
| `csrc/attention/<op>_metadata/op_kernel_aicpu/CMakeLists.txt` | 适配 csrc 的旧 AICPU 签名 |
| `csrc/attention/<op>/op_host/*_ops_base_compat.cpp` | 新增 ~56 行，补 `Ops::Base::ToString` |
| `csrc/attention/<op>/<op>_torch_adpt.h` | 新增 ~240 行，手写；**含 out 变体**（坑五） |
| `csrc/torch_binding.cpp` | +60 行，两个重载各一份 schema + impl |
| `csrc/torch_binding_meta.cpp` | +125 行，两个重载各一份 meta |
| `.pre-commit-config.yaml` | +4 个 typos 白名单词 |
| `vllm_ascend/envs.py` | +6 行开关 |
| `vllm_ascend/attention/attention_v1.py` | +447 行 |
| `vllm_ascend/compilation/acl_graph.py` | +9 行 |
