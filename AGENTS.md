# AGENTS.md

This file provides guidance to coding agents (Claude Code, Codex CLI, …) when working with code in this repository.

本仓通过符号链接把各家 agent 的配置合并到一处：`AGENTS.md` 与 `.agents/skills/` 是真身，`CLAUDE.md`、`.claude/skills` 只是指向它们的链接。要改文档或加技能，一律动 `AGENTS.md` / `.agents/skills/`。

## 这个仓库是干什么的

在昇腾 NPU 上调试与测试 **Qwen3.8**，服务由 `vllm` + `vllm-ascend` 提供。所有让模型跑起来的改动都落在 `vllm-ascend`，通过 patch / 继承 / 自定义算子等手段适配，**不改 `vllm`**。

## 硬性约束（每次动手前先确认）

1. **`vllm/` 是只读参考。** 上游 vLLM 的克隆，用来查源码、确认被 patch 的函数签名与调用点。不要修改、不要提交、不要推送。需要改上游行为时，在 `vllm-ascend` 里写 patch。
2. **本机没有 NPU，跑不了推理。** 本机只能写代码、读代码、做静态分析。有一块 5080 GPU，可以跑纯 PyTorch 的小脚本做数值等价性 / 算法逻辑的模拟验证（不能 import `torch_npu`）。
3. **验证闭环靠脚本。** 需要在真机确认权重、代码、环境或其他本机拿不到的信息时，优先把聚焦、非交互式的诊断脚本写进 `scripts/debug/`；其他真机验证脚本写进 `scripts/`。脚本要自带打印/断言和明确的 RED/GREEN 判据，输出能直接回传判读，并避免打印凭据、加载无关权重或占用 NPU，除非该验证阶段确实需要。
4. **推送只走自己的 fork。** `vllm-ascend` 的 `origin` 是本人的 fork（`TallMessiWu/vllm-ascend`，可自由推送），`upstream` 是上游官方仓（`vllm-project/vllm-ascend`）——upstream 只 fetch，永远不 push。注意这个 fork 是 public 的（fork 公开仓无法转为 private），推上去的内容对外可见。
5. **修改完成后直接提交并推送。** 在完成范围内验证后，直接生成提交信息、提交并使用普通 `git push` 推送当前分支，无需先向用户确认提交信息；主仓和子仓有改动时分别在各自仓库提交、推送。默认禁止任何强推；但如果 rebase、amend、reset 或分支历史重写使普通推送必然无法成功，应立即说明原因、目标远端分支和预期 lease，主动询问用户是否允许强推，不要为了回避询问而合并旧历史、改写方案或反复尝试无关规避手段。获得该次明确授权后，只能在刷新并核对远端分支后使用 `git push --force-with-lease`，永远禁止普通 `--force`。TLS、认证、代理等传输故障不是强推场景，合理重试后直接报告。

## 目录结构

```
qwen3.8/                   # 主仓（git，分支 main）
├── AGENTS.md              # 本文件
├── .agents/skills/        # 技能目录
├── skills-lock.json       # 外部技能来源与哈希（mattpocock/skills），由技能自身维护，勿手改
├── scripts/               # 交给用户在服务器上跑的验证/复现脚本
├── vllm/                  # submodule「vllm」→ 上游 vLLM（只读参考，当前 v0.27.2rc0+）
└── vllm-ascend/           # git worktree 根，一个分支一个目录
    ├── main/              # submodule「vllm-ascend」→ 个人 fork，跟踪 origin/main
    └── upstream-main/     # 常驻 worktree，跟踪官方 upstream/main
```

服务器上的对应源码路径是 `/home/hajimi/qwen3.8/vllm` 与 `/home/hajimi/qwen3.8/vllm-ascend/main`；容器通过 `/home:/home` 直接使用宿主机 checkout，`create-container.sh` 默认 editable 安装 `/home/hajimi/qwen3.8/vllm-ascend/feat-qfa-mxfp8-attn`（QFA 在途分支，基于 upstream/main），`main` 仍作为个人 fork 基线维护。

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
git worktree add ../feat-xxx -b feat/xxx upstream/main # 新分支
git worktree add ../bugfix-yyy origin/bugfix/yyy       # 检出已有远程分支
git worktree list
git worktree remove ../feat-xxx                        # 收尾清理
```

目录名用分支名去掉斜杠（`feat/mxfp8-quant-group-tp` → `feat-mxfp8-quant-group-tp`）。

同步上游：在 `upstream-main` 执行 `git fetch upstream && git pull --ff-only`。不要顺手把官方主线合入 `main`；`main` 保持跟踪个人 fork 的 `origin/main`，需要同步 fork 时再明确执行合并与推送。

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

`tests/ut` 里大量用例 import `torch_npu`，本机跑不了；本机改动请靠 lint + 静态阅读把关，真实验证交给服务器。

## 提交规范

**每次提交都用 `/gitmoji-commit` 技能**（已复制到 `.agents/skills/gitmoji-commit/`）：中文 subject、`<emoji-code> <type>(<scope>): <subject>` 格式。无需展示命令或等待用户确认，生成后直接提交；完成验证后再使用普通 `git push` 推送。主仓和子仓的提交都走它。

vllm-ascend 的 pre-commit 装了 `signoff-commit` 钩子，**提交必须带 sign-off**——把 `-s` 加进 gitmoji 技能生成的命令里：

```bash
git commit -s -m ":bug: fix(gdn): 修复 TP8 下 cumsum 分块导致的乱码"
```

分支命名沿用现有习惯：`feat/*`、`fix/*`、`bugfix/*`。

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

两个常驻 worktree 分别是跟踪 fork 的 `main` 和跟踪官方主线的 `upstream-main`；在途功能分支 `feat/qfa-mxfp8-attn`（QFA MXFP8 算子接入）基于 upstream/main，有独立 worktree。其他功能分支仍按任务单独创建；`scripts/` 已包含 Qwen3.8 服务启动、运行时辅助和回归测试资产，不要把这些脚本误判成插件侧适配实现。

所以别去猜「已有实现」——开新任务时先选择正确基线：fork 工作从 `main` 派生，上游工作从 `upstream-main` 派生。动某个区域前先 `git branch -r` 看看有没有相关的在途分支。
