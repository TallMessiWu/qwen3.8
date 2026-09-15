---
name: vendor-ascend-op
description: 把昇腾自定义算子接进 vllm-ascend 的阶段门禁 —— vendor 源码、写 binding、契约体检、eager 打通、进 ACL 图、做性能归因。当要 vendor 新算子(QFA 这类)、换 CANN 包后核对签名漂移、把某个算子接进图、或判断某个 kernel 能力位到底开没开时使用。
---

# 算子接入的六道门

QFA 那条线走完了全程。**顺序不能换**:跳过前一道去调后一道,拿到的现象无法归因。

## 门 1 — vendor 源码

- 从官方 ops-transformer 原样拷进 `csrc/attention/`,**各自内嵌 attention/common 依赖头**。
- ⛔ **永远不要覆盖 csrc 共享的 `common/`**,那是旧裁剪快照,覆盖会连累别的算子。
- 伴生算子(主算子 + AICPU metadata 算子)**必须成对,两次调用参数逐项一致**。
- ⛔ **不改算子本体**(`op_api/` `op_host/` `op_kernel/` `common/`)——这是硬约束,也不是一个人的工作量。
- 记下 vendor 的上游 commit。上游后续改动不会自动跟进,要人工比对。

## 门 2 — binding

- **主动出 `.out` 变体**,哪怕现在用不上:aclgraph 要它(aclnn 本来就收 `attnOut`/`softmaxLse`),
  是纯 binding 层的事,不碰 kernel。三处要加:`*_torch_adpt.h` 薄封装、`torch_binding.cpp` schema、
  `torch_binding_meta.cpp` meta。
- **out-of-tree backend 的 `get_name()` 必须返回 `"CUSTOM"`**:`Attention.__init__` 做
  `AttentionBackendEnum[get_name()]`,闭合枚举,插件侧唯一槽位就是 CUSTOM。backend 身份靠
  `platform.get_attn_backend_cls()` 区分。(`AscendMLABackend` 返回 `"ASCEND_MLA"` 不炸是因为 MLA 不走这条查表,别照抄。)
- 开关走 `vllm_ascend/envs.py` 的 `env_variables`,命名 `VLLM_ASCEND_*`;
  **在 `__init__` 里读一次存成 `self.xxx`,热路径不能每步 `os.getenv`**。

## 门 3 — 契约体检

```bash
python scripts/checks/qfa_op_contract.py     # 换 CANN 包后先跑,比对签名漂移
```

换 CANN、换 vendor 版本、rebase vendor 之后**先跑这个再谈别的**。签名漂了而不知道,
后面每一个现象都会被归因错。

## 门 4 — eager 打通

- 单算子先跑 golden 对拍(带 case 的那种),再谈端到端。
- **精度要三方对比**,把量化损失和算子计算差异拆开:新算子 vs 参考算子(bf16)、
  新算子 vs 参考算子(喂反量化后的同输入)。QFA 的分解是
  `sqrt(5.30² − 2.65²) ≈ 4.6%` 是 K/V 压 MXFP8 的信息损失,2.65% 是 fp8 matmul 相对 bf16 的计算精度差
  ⇒ 是算子固有特性,不是接入 bug。
- ⚠️ **bench 脚本现算的「最优」scale 给出的是下界**,框架用 checkpoint 里的静态 per-channel scale,
  只会更差;bench 传紧的 `max_seqlen_kv` 而框架传常量,框架的 tiling 只会更保守。别拿 bench 数当端到端承诺。
- 精度的前提是权重侧先干净:scale 有没有真读进来,别在走兜底值的时候谈精度(只能谈通不通、崩不崩、省多少显存)。

## 门 5 — 进图

**验证捕获真的发生,看日志三行**,`serves=True` 证明不了(它在 eager 路径也打,乌龙过一次):

- `captured op #N into task group at tokens=S` —— 只有拿到 handle 才可能打
- `update tokens=S -> updating {'ops': N, ...}` —— ops 必须等于全注意力层数;出现 `skip: nothing captured` 就是没捕上
- `plan #N ... header=(...)` —— 捕获期与 replay 期必须相同

进图的两条路,**选一条别混**:每次 replay 重绑整个调用(FIA 的 `graph_task_update` 是「整个调用重跑一遍」),
或读捕获期自己持有的 buffer(QFA 走后者,不需要 `graph_task_update`)。

**流依赖**:builder 流上跑 AICPU metadata、update_stream 上 `copy_` 结果,两条流无依赖 ⇒ 拷贝可能跑赢 AICPU 的写,
捕获的算子读到半成品 plan。⛔ **不能 `update_stream.wait_stream(current_stream)`——图已入队并停在事件上,会死锁**;
正解是 plan 入队后 `record_event()`,`update_stream.wait_event()` 只等那一点。

**进图后只能用常量的东西**:被烘进捕获算子的参数(如 `max_seqlen_kv`)必须在捕获期与 replay 期同值,
所以取 `max_model_len` 这类常量。metadata 算子与主算子必须同值。

**MTP 多步进同一张图不是障碍**:把 `(draft_step, key)` 展平成有序列表、按捕获顺序逐个对齐即可(FIA 一直这么做)。
每步的输入张量必须各自分配——proposer 先把所有 draft 步的 metadata 建完再逐步跑,共用一份 builder buffer
会让所有 draft 步读到最后一步的值。

**验证要跑足够多的 decode 步**,别被探针前几次的 D2H 意外串行化掩盖住竞态。

其余图捕获的时刻陷阱见 `graph-capture-timing`。

## 门 6 — 性能归因

**先确认能力位是不是编译期就关掉的,再谈调优。** QFA 的 flash-decode 是 host tiling
(`flashDecodeFlag_ = false`)、kernel(`constexpr bool isFdConst = false` → `if constexpr` 整段剪掉)、
AICPU(`param.fdOn = 0`)**三处配套关闭**,不是一行疏忽。

⚠️ **模板实参 ≠ 代码被生成**:`VecFdBlock` 确实作为模板实参传进了 Kernel,但所有调用点都在
`if constexpr(false)` 内 ⇒「实例化了所以链路通」是误读。

⛔ **「只改一处自证」的实验不能做,做了必错**:只改 AICPU 会让 plan 排出 FD 任务,而 kernel 里没有代码
消费它们、workspace 也没分配 ⇒ 拿不到加速,只会得到错误结果或非法访问。

性能测量要拆维度扫(prefill 长度 × decode batch × KV 长度),单点数会骗人:QFA 是
**prefill 越长越赢(1.40x→2.45x)、decode 全线落败(0.24x~0.55x),唯独 b1-128k 是 1.47x**,
根因就是 FD 关着、decode 并行度只能来自 `batch × heads`。

端到端换算要乘全注意力层数、metadata 每步只付一次。

## 文档在哪

- ops-transformer 仓只有 kernel 实现 + README,**接口文档在别处**。
- 算子官方文档走 `raw.gitcode.com` 可直连 `curl`(`gitcode.com/.../raw` 只返回 HTML 壳)。
- 真机行为与文档冲突时以真机为准,并把实测结论写回记忆(带日期和 file:line)。

## 数据搬运

NPU 上 **FP8 张量必须用字节视图搬运**:`index_put_` / `transpose` 对 float8 会报错或回退 AICPU 打死 device,
一律 `.view(torch.uint8)` 搬完再换回。
