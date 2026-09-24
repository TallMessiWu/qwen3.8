# AGENTS.md

This file provides guidance to coding agents (Claude Code, Codex CLI, …) when working with code in this repository.

本仓通过符号链接把各家 agent 的配置合并到一处：`AGENTS.md` 与 `.agents/skills/` 是真身，`CLAUDE.md`、`.claude/skills` 只是指向它们的链接。要改文档或加技能，一律动 `AGENTS.md` / `.agents/skills/`。

## 这个仓库是干什么的

在昇腾 NPU 上调试与测试 **Qwen3.8**，服务由 `vllm` + `vllm-ascend` 提供。所有让模型跑起来的改动都落在 `vllm-ascend`，通过 patch / 继承 / 自定义算子等手段适配，**不改 `vllm`**。

## 硬性约束（每次动手前先确认）

1. **`vllm/` 是只读参考。** 上游 vLLM 的克隆，用来查源码、确认被 patch 的函数签名与调用点。不要修改、不要提交、不要推送。需要改上游行为时，在 `vllm-ascend` 里写 patch。
2. **本机没有 NPU，跑不了推理。** 本机只能写代码、读代码、做静态分析。有一块 5080 GPU，可以跑纯 PyTorch 的小脚本做数值等价性 / 算法逻辑的模拟验证（不能 import `torch_npu`）。本机已有一套对齐真机的 uv 环境（`.venv/`，见 `scripts/local/`），推真机之前先用它跑 patch 目标体检和 CPU 单测。
3. **验证闭环靠脚本。** 需要在真机确认权重、代码、环境或其他本机拿不到的信息时，写聚焦、非交互式的诊断脚本。脚本要自带打印/断言和明确的 RED/GREEN 判据，输出能直接回传判读，并避免打印凭据、加载无关权重或占用 NPU，除非该验证阶段确实需要。
   **按存活周期选目录**：只为回答当下这一个问题的写进 `scripts/debug/`，问题一结案就连脚本一起删（git 留着历史；过期的诊断仍然跑得动、输出仍然像回事，比没有更坏）。会被反复重跑的按类别归位——算子精度与性能进 `scripts/bench/`，权重与设备体检进 `scripts/checks/`，构建安装进 `scripts/setup/`。判据是「下次换权重 / 换 CANN / rebase vendor 时还会不会再跑它」。
   **产出只写脚本执行目录的相对路径**：`/tmp` 会挤爆服务器本就紧张的根分区；`~`、`$HOME`、`/home/<user>/` 这类绝对家目录路径同样不许出现在交给用户跑的命令里（这条被纠正过两次）。给用户的命令写成 `| tee qfa.log`，后续 `grep` 也用相对路径；产物落在仓目录里时顺带提醒用完 `rm` 掉。
4. **推送只走自己的 fork。** `vllm-ascend` 的 `origin` 是本人的 fork（`TallMessiWu/vllm-ascend`，可自由推送），`upstream` 是上游官方仓（`vllm-project/vllm-ascend`）——upstream 只 fetch，永远不 push。注意这个 fork 是 public 的（fork 公开仓无法转为 private），推上去的内容对外可见。
5. **修改完成后直接提交并推送。** 在完成范围内验证后，直接生成提交信息、提交并使用普通 `git push` 推送当前分支，无需先向用户确认提交信息；主仓和子仓有改动时分别在各自仓库提交、推送。默认禁止任何强推；但如果 rebase、amend、reset 或分支历史重写使普通推送必然无法成功，应立即说明原因、目标远端分支和预期 lease，主动询问用户是否允许强推，不要为了回避询问而合并旧历史、改写方案或反复尝试无关规避手段。获得该次明确授权后，只能在刷新并核对远端分支后使用 `git push --force-with-lease`，永远禁止普通 `--force`。TLS、认证、代理等传输故障不是强推场景，合理重试后直接报告。
6. **改主仓文件必须保持 LF 行尾。** 主仓 `core.autocrlf=false`（工作区存什么就提交什么），而 `vllm-ascend` 子仓是 `true`（git 自动规范化，怎么写都干净）。用 Python 读-改-写主仓文件时，`Path.write_text()` / `open(..., "w")` 默认把 `\n` 翻成 `\r\n`，一次三行的改动会变成整篇重写的 diff，review 和 blame 全废。写文件统一传 `newline="\n"`，改完用 `file <path>` 确认没有 "with CRLF line terminators"，并和同目录其他文件比对。
7. **合入别人的分支只做必要改动。** 把同事或上游的分支合进来做适配时，只改「不改就跑不通」的行：不顺手重构、不重写 docstring、不整理测试结构、不加防御性代码。「现在没人调」不等于「可以删」，那可能是对方为另一条路线留的。改完用 `git diff <他们的提交> -- <文件>` 逐文件复查，凡是解释不出「不改会怎样」的行就退回去；必要的改动在提交信息或注释里写清被迫的原因。改动越多，将来和对方分支再同步越难，责任边界也糊了。

## 目录结构

```
qwen3.8/                   # 主仓（git，分支 main）
├── AGENTS.md              # 本文件
├── .agents/skills/        # 技能目录
├── docs/                  # 文档：reference/ 反复要查的事实，cases/ 一次排查的记录（总目录见 docs/README.md）
├── skills-lock.json       # 外部技能来源与哈希（mattpocock/skills、tt-a1i/archify），由 skills CLI 维护，勿手改
├── pyproject.toml         # 本机 uv 环境的依赖清单（venv 在 .venv/，不进主仓）
├── scripts/               # 交给用户在服务器上跑的验证/复现脚本
│   ├── bench/             # 算子精度与性能基准，长期复用
│   ├── checks/            # 权重与设备体检，长期复用
│   ├── setup/             # 构建与安装入口
│   ├── debug/             # 一次性诊断的暂存区，查完即删
│   └── local/             # 例外：只在本机跑的环境搭建与静态检查脚本
├── vllm/                  # submodule「vllm」→ 上游 vLLM（只读参考，跟随 vllm-ascend 的 .github/vllm-main-verified.commit）
└── vllm-ascend/           # git worktree 根，一个分支一个目录
    ├── main/              # submodule「vllm-ascend」→ 个人 fork，跟踪 origin/main
    └── upstream-main/     # 常驻 worktree，跟踪官方 upstream/main
```

服务器上的对应源码路径是 `/home/hajimi/qwen3.8/vllm` 与 `/home/hajimi/qwen3.8/vllm-ascend/main`；容器通过 `/home:/home` 直接使用宿主机 checkout，`create-container.sh` 默认 editable 安装 `/home/hajimi/qwen3.8/vllm-ascend/junlin-c8-mxfp`（C8 MXFP8 在途分支，基于上游 PR 15484），`main` 仍作为个人 fork 基线维护。

**主仓只跟踪两个 submodule 指针 + `scripts/` + agent 配置。** `vllm-ascend/main` 之外的 worktree 目录（包括 `upstream-main` 和各任务分支）被 `.gitignore` 排除（`/vllm-ascend/*` + `!/vllm-ascend/main`），留在本地不进主仓。

submodule 指针只有在需要固定「这套脚本对应哪个版本的 vllm / vllm-ascend」时才更新，日常在子仓里提交不必顺手 bump：

```bash
git submodule status                      # 看两个子仓当前指向的 commit
git add vllm-ascend/main && git commit    # 需要时才推进指针
git submodule update --init --recursive   # 新机器克隆后拉起子仓
```

## Worktree 工作流

两个常驻 worktree 分别维护 fork 基线与官方主线；先各自快进到对应远端：

```bash
git -C vllm-ascend/main pull --ff-only                 # origin/main
git -C vllm-ascend/upstream-main pull --ff-only        # upstream/main
```

每个任务仍使用独立分支和目录。面向上游的新改动通常从 `upstream-main` 派生：

```bash
cd vllm-ascend/upstream-main
git worktree add ../feat-xxx -b feat-xxx upstream/main # 新分支
git worktree add ../bugfix-yyy origin/bugfix-yyy       # 检出已有远程分支
git worktree list
git worktree remove ../feat-xxx                        # 收尾清理
```

分支名不用斜杠，直接 `feat-xxx`、`bugfix-yyy` 这种扁平写法，目录名与分支名保持一致。

同步上游：在 `upstream-main` 执行 `git fetch upstream && git pull --ff-only`。不要顺手把官方主线合入 `main`；`main` 保持跟踪个人 fork 的 `origin/main`，需要同步 fork 时再明确执行合并与推送。

## 本机验证环境（推真机之前先过一遍）

`scripts/local/setup-devenv.sh` 在本机建一套对齐真机的 venv：Python 3.11 +
vllm（直接装 `vllm/` submodule 当前的 checkout，`VLLM_TARGET_DEVICE=empty`，不编译 kernel）
+ torch 2.10.0（带 CUDA，吃 5080）+ editable 的 `vllm-ascend/junlin-c8-mxfp-16278`。`torch_npu` 不装，由 vllm-ascend 自己的
`tests/ut/conftest.py` 在探测不到 `npu-smi` 时注入 MagicMock——跟 CI 的 CPU runner 同一口径。

```bash
bash scripts/local/setup-devenv.sh              # 建/重建环境
python scripts/local/check_patch_targets.py     # patch 目标还在不在、参数列表有没有漂
bash scripts/local/run_cpu_ut.sh                # CPU 单测 + 跟已知基线比对
```

venv 的 vllm 直接装 `vllm/` submodule 当前的 checkout，它跟着 vllm-ascend 的
`.github/vllm-main-verified.commit` 走（当前 `84030bbe3d`）。真机是 pip 装的 0.28.0；
**这两者不等价**：本机自报 `0.28.1rc1.dev676+g84030bbe3`，于是 101 处
`vllm_version_is("0.28.0")` 守卫在本机全为 False、在真机全为 True，本机跑的是另一条
lane。`VLLM_VERSION=0.28.0` 强行对齐会在 collection 就炸（0.28.0 分支要
`vllm.model_executor.layers.attention.pcp`，main 已挪到 `vllm.v1.attention.ops.pcp`），
要验 0.28.0 那条分支只能覆盖 `VLLM_REF` + `VLLM_WORKTREE` 从 submodule 临时派生一个
只读 worktree 重建 venv（用完 `git -C vllm worktree remove` 删掉）。

**⚠️ 2026-09-15 起 0.27.1 已经不能用了。** 近期 upstream/main 的 vllm-ascend 只支持
**vllm 恰好 0.28.0**（代码里 101 处 `vllm_version_is("0.28.0")` 守卫）或 **main
`84030bbe3d`**；0.27.1 会在 `patch_kv_cache_utils.py` 就因 `_get_packed_kv_cache_groups`
缺失而 import 失败。`junlin-c8-mxfp` / `junlin-qfa` 这两条老分支还钉着 `ba07e4a48f`，
要验它们得先把 `vllm/` 切过去再重建 venv——换版本重建会清掉 `.venv`，两套环境没法并存。
（曾经 `vllm/` 停在 v0.28.1rc0、领先真机 1300+ commit，照着它查会得出对不上的结论；
现在方向反过来了，同样要小心。）
能拦什么、拦不住什么、已知的基线失败（当前 1 条），见 `scripts/local/README.md`。

## 常用命令（全部在某个 vllm-ascend worktree 目录内执行）

```bash
# Lint / 格式化：提交前必跑，覆盖所有文件类型（含 markdown）
bash format.sh          # 等价 pre-commit run --all-files
bash format.sh ci       # CI 口径，含 manual stage 的钩子
pre-commit run ruff-check --all-files    # 只跑单个钩子

# 单元测试
pytest -sv tests/ut/ops/test_prepare_finalize.py
pytest -sv tests/ut/ops/test_prepare_finalize.py::test_prepare_inputs

# e2e（需要 NPU 硬件，只能在服务器上跑）
pytest -sv tests/e2e/pull_request/one_card/aclgraph/test_aclgraph_accuracy.py
```

`format.sh` 在**本机**跑有四个坑（2026-09-14 逐个踩过）：

1. **pre-commit 没装，且 `.venv` 里没有 pip**（uv 建的，`python -m pip` 直接报 No module named pip）。装法：`uv pip install --python .venv/bin/python pre-commit`，再把 `.venv/bin` 放进 `PATH` 后才跑 `format.sh`。
2. **`--all-files` 会连带重排与本次改动无关的文件。** 至少 `vllm_ascend/utils.py`、`vllm_ascend/attention/attention_v1.py`、`vllm_ascend/quantization/configs/modelslim_config.py`、`tests/ut/quantization/methods/test_mxfp_c8.py` 四个在 HEAD 状态下就不符合 ruff v0.14.0（纯行宽重排）。**跑完必须 `git checkout --` 还原它们**，否则会把无关的全文件格式改动带进 commit。所以 `ruff format` 报 Failed 不等于自己的代码有问题，先看它改了哪些文件。
3. **别用 venv 里的 ruff 下结论。** venv 是 0.16.5，`.pre-commit-config.yaml` 钉的是 v0.14.0，两者的格式判断不一致。验自己改的文件要走 pin 的那份：`pre-commit run ruff-format --files <files>` / `pre-commit run ruff-check --files <files>`。
4. **shellcheck 钩子首次跑会假失败。** 它把二进制下载解压到 worktree 下的 `shellcheck-stable/`（在 `.gitignore` 里），并行解压同一个文件会报 `Text file busy`；二进制落地后重跑即过，用完可以删掉。

`tests/ut` 大量用例 import `torch_npu`，但 `tests/ut/conftest.py` 探测不到 `npu-smi` 时会注入 MagicMock，所以本机能跑（走 `scripts/local/run_cpu_ut.sh`，当前 4374 passed / 1 failed）。e2e 和真实数值仍然只能上服务器。

## 仓库专属技能（`.agents/skills/`）

除了通用技能，这三个是本仓踩出来的，开工前按场景挑：

- **`npu-remote-diagnose`** —— 真机故障排查开局。反馈循环在服务器上、一轮几分钟，
  所以纪律是「每轮排除最多假设」：单变量开关矩阵、哑开关检查、探针纪律（warning 级 /
  地基探针 / 去重键）、设备侧指纹二分、排除台账与证据分级。通用的 `diagnosing-bugs`
  假设本机能建 2 秒的紧循环，那个前提在这里不成立。
- **`graph-capture-timing`** —— 「值在错误的时刻被固定」这一族：编译区里的 Python 求值被烘死、
  ACL 捕获只记录不执行、捕获期与 replay 期地址不同。长 prompt 吐 EOS 和 MTP 接受率塌都是它。
  改任何进图代码前先过一遍，配套脚本 `scripts/checks/scan_capture_timing.py`。
- **`vendor-ascend-op`** —— 算子接入的六道门：vendor / binding / 契约体检 / eager / 进图 / 性能归因。
  换 CANN 包、vendor 新算子、把算子接进图时用。

## 提交规范

**每次提交都用 `/gitmoji-commit` 技能**（已复制到 `.agents/skills/gitmoji-commit/`）：中文 subject、`<emoji-code> <type>(<scope>): <subject>` 格式。无需展示命令或等待用户确认，生成后直接提交；完成验证后再使用普通 `git push` 推送。主仓和子仓的提交都走它。

vllm-ascend 的 pre-commit 装了 `signoff-commit` 钩子，**提交必须带 sign-off**——把 `-s` 加进 gitmoji 技能生成的命令里：

```bash
git commit -s -m ":bug: fix(gdn): 修复 TP8 下 cumsum 分块导致的乱码"
```

分支命名不用斜杠，用连字符扁平写法：`feat-xxx`、`fix-xxx`、`bugfix-xxx`。不再使用的分支改名为 `archive-xxx` 归档，不直接删除。

## vllm-ascend 架构要点

**它是 vLLM 的硬件插件，不是 fork。** 通过 `setup.py` 的 entry_points 注册：

- `vllm.platform_plugins`: `ascend = vllm_ascend:register`
- `vllm.general_plugins`: KV connector / model loader / service profiling / model 注册

**Patch 分两个阶段生效**，选错阶段会导致 patch 不生效或在错误进程里打：

| 阶段 | 目录 | 触发点 |
| --- | --- | --- |
| platform | `vllm_ascend/patch/platform/` | worker 启动前，`NPUPlatform.pre_register_and_update()` → `adapt_patch(is_global_patch=True)` |
| worker | `vllm_ascend/patch/worker/` | 每个 worker 的 `__init__`，`adapt_patch(is_global_patch=False)` |

新增 patch **必须**在 `vllm_ascend/patch/__init__.py` 追加说明块，四段齐全：Why / How / Related PR（没有就解释为什么没有）/ Future Plan。这是 review 硬要求。

**改动优先级**：patch < 继承 < 直接改 model_runner。能用 patch 就别动 model_runner——`vllm_ascend/worker/model_runner_v1.py`（v1）、`vllm_ascend/worker/v2/model_runner.py`（v2）、`vllm_ascend/_310p/model_runner_310p.py`（310P）的改动都需要架构级 review。

**设备差异走 `vllm_ascend/device/device_op.py`**：`BaseDeviceAdaptor` + 各代芯片子类（A2/A3/A5/310P，见 `AscendDeviceType`）。某代芯片缺算子时在这里做 Triton / 原生回退，而不是在调用处写 if-else。

**环境变量集中在 `vllm_ascend/envs.py`** 的 `env_variables` 字典里，命名 `VLLM_ASCEND_*`，用 `from vllm_ascend import envs` 引用，禁止散落硬编码字符串。新增变量需要 review。

**NPU 性能红线**：设备张量上的 `tensor.item()` 会触发 NPU→CPU 同步，热路径里会卡住 `AsyncScheduler`。优先保持数据在设备侧（`torch.max` / `torch.argmax` 等），必须同步时合并成一次批量同步并写注释说明原因。

完整规范见 `vllm-ascend/main/AGENTS.md`（代码风格、测试要求、review checklist）。

## 当前状态

两个常驻 worktree 分别是跟踪 fork 的 `main` 和跟踪官方主线的 `upstream-main`。在途分支各有独立 worktree：

- `junlin-c8-mxfp` —— 跟随上游 PR 15484（C8 MXFP8 KV cache + QFA + MTP + PD 分离），基于该 PR 头部。容器和本机 venv 默认 editable 安装的就是它。它的 QFA 来自外部 `cann_ops_transformer` 包，csrc 里没有算子源码，也没有 `VLLM_ASCEND_ENABLE_QFA` 开关。
- `junlin-qfa` —— QFA 算子接入主线，基于 upstream/main，官方 master QFA 已 vendor 进 csrc。`scripts/setup/` 下三个 `*qfa*` 构建脚本仍默认指向它，因为只有它能从 csrc 编出 QFA。
- `junlin-c8-mxfp-16614` —— 从上游 PR 16614 头部（本地快照 `pr-16614`）派生。PR 16614 就是 `junlin-c8-mxfp-16278` 那 10 个提交被 rebase 到更新的上游，补丁内容一致。在它之上多两个提交，修的是 **hybrid C8 的 KV 容量只有 BF16 的四分之一**（2026-09-21，仅本机验证，真机待验）：mamba 的配置钩子跑在 `quant_config` 赋值之前，看不到 C8，按 BF16 把 page 定成「2048 token」那么大；随后 `refresh_block_size` 又把 block 写死成 512，于是每个 attention page 有 87% 是填充。现在 hybrid 下按 FP8 字节重算——block 取「一个 SSM state 能装下的 FP8 token 数」（2.4T@TP8 是 4096 = 8 个 512 的 QFA kernel 块），K 段与 ssm 段逐块重合，和 BF16 路径同一个不变量；不能整除时直接报错，凑整会让两组的 block id 串扰。配套把 prefix-cache 的 CoW 块拷贝改成按 kernel 行展开。`junlin-c8-mxfp`、`junlin-c8-mxfp-16278`、`junlin-qfa` 都还带着写死 512 的旧逻辑，这两个提交能无冲突 cherry-pick 过去。细节（真机验收预期值、从「GPU KV cache size」反推 `num_blocks` 的方法）见 `docs/cases/hybrid-c8-kv-capacity.md`。

前两条分支各自带着同样的两个真机故障修复，都是「值在错误的时刻被固定」这一类：

1. **MoE 三处 TP 规约按当前通信方式重算**。`ALLGATHER` 是唯一不在融合 kernel 里做 TP 规约的通信方式，而通信方式随每步 token 数变化；原来那个判据在编译区里求值一次、被烘成捕获期 dummy run 的值，长 prompt 下 all_reduce 整个没执行，模型直接吐 EOS。三个消费点必须同时改，只改一处会让 shared 被规约两次、从吐 EOS 变成整段乱码。
2. **C8_MXFP 的 V scale 缓存不能在捕获期标记为已填充**。ACL 图捕获只记录不执行，而记录「填过了」的那行 Python 是真执行的，于是 V 的 scale 缓存永远全零、反量化后 attention 恰好吐零。target 不受影响，draft 第一次走到这段就是捕获本身，表现为一开 draft 图 MTP 接受率就塌。

两条分支的 QFA 进图机制不同，排查时不要互相套用结论：`junlin-c8-mxfp` 是原生 `npugraph_ex`，`junlin-qfa` 是 task group + update 重发。

2026-09-10 清理过一轮分支。`junlin-qfa-c8switch`（C8 开关拿 bf16 基线）、`feat-qfa-dump`（QFA 输入输出 dump 插桩）、`debug-moe-comm-tokens`（MoE 通信判据打印）以及全部 `archive-*` 备份都已从本地和 fork 删除，远端只剩 `main`、`junlin-qfa`、`junlin-c8-mxfp` 三条。`scripts/bench/replay_qfa_dump.py` 的配套插桩（`vllm_ascend/attention/qfa_dump.py`）随 `feat-qfa-dump` 一起没了，回放脚本因此跑不起来，同日一并删除；真要重做 QFA dump，捕获侧和回放侧得一起写回来。

其他功能分支仍按任务单独创建；`scripts/` 已包含 Qwen3.8 服务启动、运行时辅助和回归测试资产，不要把这些脚本误判成插件侧适配实现。

所以别去猜「已有实现」——开新任务时先选择正确基线：fork 工作从 `main` 派生，上游工作从 `upstream-main` 派生。动某个区域前先 `git branch -r` 看看有没有相关的在途分支。
