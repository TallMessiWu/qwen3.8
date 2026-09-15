---
name: npu-remote-diagnose
description: 昇腾真机故障的排查纪律 —— 反馈循环不在本机、一轮实验要用户上服务器跑几分钟,所以优化目标是「每轮排除最多假设」而不是「循环跑得快」。当出现空输出 / 乱码 / 接受率塌 / 崩溃 / 性能不及预期,要设计单变量实验、写诊断脚本、布探针、或判断某个结论到底站不站得住时使用。
---

# 真机排查:一轮很贵

通用的 `diagnosing-bugs` 假设你能把反馈循环压到 2 秒。**这里做不到**:本机没有 NPU,
每一轮实验要写脚本 → 用户上服务器跑 → 回传输出,几分钟到几十分钟一轮。所以纪律不同:
不追求循环快,追求**每一轮排除掉最多的假设**,以及**不白跑**。

历史上白跑的轮次全部来自下面这几类,不是来自想得不够多。

## 开局:先把复现条件收敛成最小开关集

`scripts/` 里那批 `27B.sh` / `397B.sh` 的开关(`GRAPH` / `MTP` / `QFA` / `C8` / `EP` /
`SPEC_EAGER` / `COMPILE_MODE` / `ASYNC` / `TP`)就是为这件事存在的。先出一张合取表,例如:

| EP | GRAPH | ≤400 token | >400 token |
|---|---|---|---|
| 1 | 1 | ✓ | ✗ |
| 1 | 0 | ✓ | ✓ |
| 0 | 1 | ✓ | ✓ |

三条件缺一不可 ⇒ 后面所有假设都必须能同时解释这三条边。**不能解释全表的假设直接丢掉,别测。**

### 开关必须是干净单变量

- `SPEC_EAGER=1`(`speculative_config.enforce_eager`)**只关 draft 图**,target 照常 FULL 捕获,
  且不回灌 target 的 CompilationConfig —— 这是区分 target/draft 的干净刀。
- `GRAPH=0` / `CUDAGRAPH_MODE=PIECEWISE` 同时改了 target;**PIECEWISE 下 draft 根本不进图**
  (`has_full_cudagraphs()` 为假),所以「PIECEWISE 正常」推不出任何关于后端的结论。
- 调 `MAX_NUM_SEQS` / `MTP` 会让 MC2 容量两边一起变,**不能靠它分离变量**。
- 改 `CompilationConfig._attention_ops` 让某个 op 单独回图里,这个 hack **自身有副作用**
  (两个方向都乱码且接受率更低),拿它得到的结论不可用。

### 开关是不是哑的

用某个开关前先确认它在**当前分支**真的被定义。`C8=0`
(`VLLM_ASCEND_DISABLE_C8_MXFP`)在 `junlin-qfa` 上是哑开关——只在另一条分支定义,
设了等于没设,一整轮实验白跑。`grep` 一下 `vllm_ascend/envs.py` 再用。

## 探针纪律

1. **`logger.warning` 级,不是 info。** info 级探针在真机日志里一次都没出现过,白跑两轮无法判断代码到底有没有被调用。
2. **模块导入时打一行版本标记。** 否则分不清「探针没触发」和「新代码没装上」。
3. **留一条地基探针**——无论走哪条路都打的那条(`[moe-sel]` 无论选谁都打)。其余探针都挂在
   具体某条路径上,压根没选中时它们一起为 0,与「插桩失效」无法区分。已因此白读过一轮日志。
4. **去重键必须含你要区分的维度。** `[qfa] capture` 按 `(state, serves)` 去重,只打一行**不代表**
   只走过一种长度。
5. **editable 安装改了代码必须重启服务。**
6. 前向里不能直接 log:会图断裂,捕获期还非法。要采数据用**设备侧指纹**——`copy_` 累到一块设备张量上,
   跑完统一读一次。

## 二分:采同一组指纹,逐层收窄

定位 MTP 接受率那次真正起作用的是这个:两侧(eager / graph)同时采**同一组设备侧指纹**,对前向做二分。
先 proposer 层 → 再 MTP head 子块(embed/norm/fc/layer/out)→ 再 attention 内部(q/k/原始输出/投影)
→ 落到「q、k 逐位相同、输出恰好为零」。

**逐位相同本身就是信息**:说明你改的东西不在这条路径上。此前有九次「猜某个描述符不对 → 改了跑」,
结果全部逐位相同。**连续两三次逐位 no-op 之后就该换方法,而不是继续猜下一个描述符。**

## 排除台账

每条「已排除」都要记明**凭什么排除**,否则下一轮会重测,或者更糟——把真因写进排除清单。

⛔ **「只有一条分支有 / 只有一处代码碰它」不是排除依据。** `_v_scale_filled_caches` 就是被这么
错误地写进「已排除,别重做」的,它恰恰是真因。

⛔ **本机没复现不是排除依据。** 本机 `torch.compile` 三种写法都测不出真机那次折叠。

结论按证据强度分级,写进记忆/笔记时带上级别:

| 级别 | 例 |
|---|---|
| 真机实测 | 能贴日志行 / 退出码 |
| 静态读码 | 能给 `file:line`,且三处(host tiling / kernel / AICPU)互相自洽 |
| 本机模拟 | 只能证伪算法逻辑,证不了运行期行为 |

## 归因陷阱

- **AICPU / 异步下发会骗人**:下发是异步的,traceback 指向的是当时在等的那个算子。
  真信息在 plog(`/root/ascend/log/run/plog/`,不是 debug/plog),找 `fault kernel_name=` 和
  `EE9999 ... rtEventRecord execution failed`。定位靠在阶段之间插同步,不是读 traceback。
- **指标语义**:`vllm:spec_decode_num_accepted_tokens_per_pos_total` 是**生存曲线**不是逐位独立接受率
  (`observe_draft` 对 `range(num_accepted)` 全部 +1)。用 `scripts/checks/mtp_accept_rate.py` 取绝对计数,
  别信日志里的两位小数比率。
- **验证要跑够步数**:探针每 shape 只打前 3 次时,竞态窗口在那之后才打开;decode 步数不够会把 bug 盖住。
- **一个函数名可能对应两种完全不同的故障**,只看错误码别看名字。

## 脚本交付

按 `AGENTS.md` 硬性约束 3 写:非交互、自带 RED/GREEN 判据、输出能直接回传判读、不打印凭据、
不写 `/tmp`(服务器根分区紧张,写脚本执行目录的相对路径,用完删)。

**退出码单独取,别用管道接 `tail`,也别信包装脚本的返回值**——已经翻车两次:红了照样提交推送。

```bash
bash some_check.sh > out.log 2>&1; echo "exit=$?"; tail -30 out.log
```

按存活周期选目录:只为回答当下这个问题的进 `scripts/debug/`,**结案连脚本一起删**;会反复重跑的
按类别进 `scripts/bench/`(算子精度与性能)、`scripts/checks/`(权重与设备体检)、`scripts/setup/`(构建安装)。

## 推真机之前

```bash
python scripts/local/check_patch_targets.py     # patch 目标还在不在、参数有没有漂
bash scripts/local/run_cpu_ut.sh                # CPU 单测 + 跟已知基线比对
```

⚠️ 本机 venv 一次只对得上一条 vllm lane,`vllm_version_is("0.28.0")` 守卫本机恒 False、真机恒 True,
本机跑的是另一条 lane。详见 `scripts/local/README.md`。
