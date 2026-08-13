# qwen3.8

在昇腾 NPU 上调试与测试 **Qwen3.8** 的工作区。服务栈是 `vllm` + `vllm-ascend`，所有适配改动只落在 `vllm-ascend`。

## 这个仓库为什么存在

它本身不含推理代码，是一个**给 AI coding agent 用的工作区壳子**——把「让 agent 高效开发 vllm-ascend 上的 Qwen3.8」需要的东西聚到一处：

- **上下文完整**：`vllm/`（上游只读参考）和 `vllm-ascend/`（实际开发的私仓）以 submodule 挂在同一棵树下，agent 一次 clone 就能同时读到「被 patch 的上游函数长什么样」和「patch 怎么写的」，不用在两个互不可见的仓库之间猜。
- **规则前置**：`AGENTS.md` 写清硬性约束（`vllm/` 只读、本机无 NPU、patch 分阶段、性能红线等），Claude Code 与 Codex CLI 通过符号链接读同一份，不会各说各话。
- **技能复用**：`.agents/skills/` 收着提交规范（`gitmoji-commit`）、调试（`diagnosing-bugs`）、review、TDD、研究等技能，两家 agent 共用。
- **验证闭环**：本机没有 NPU，跑不了推理。需要真机验证时把可复现脚本写进 `scripts/`，推到主仓，由人在服务器上执行并回传输出。`scripts/` 就是这条回路的交接点。

想直接动手写代码，先读 [`AGENTS.md`](AGENTS.md)——那里是给 agent 的完整规则，本文件只负责讲清楚仓库怎么搭起来、怎么拉下来。

## 目录结构

```
qwen3.8/                   # 主仓（git，分支 main）
├── README.md              # 本文件
├── AGENTS.md              # agent 规则真身（约束 / 架构要点 / 提交规范）
├── CLAUDE.md              # → AGENTS.md（符号链接）
├── .agents/skills/        # 技能目录真身
├── .claude/skills         # → .agents/skills/（符号链接）
├── skills-lock.json       # 外部技能来源与哈希，由技能自身维护，勿手改
├── scripts/               # 交给用户在服务器上跑的验证/复现脚本
├── vllm/                  # submodule → 上游 vLLM（只读参考）
└── vllm-ascend/           # git worktree 根，一个分支一个目录
    └── main/              # submodule → 私仓，主 worktree，跟踪 origin/main
```

要改规则或加技能，一律动 `AGENTS.md` / `.agents/skills/`，别去改那两个符号链接。

主仓只跟踪 **两个 submodule 指针 + `scripts/` + agent 配置**。`vllm-ascend/main` 之外的 worktree 目录被 `.gitignore` 排除（`/vllm-ascend/*` + `!/vllm-ascend/main`），留在本地不进主仓。

服务器上的对应路径是 `/vllm-workspace/vllm` 与 `/vllm-workspace/vllm-ascend`（见 `vllm-ascend/main/Dockerfile`），写脚本时按服务器布局写路径。

## 克隆

### 一步到位

```bash
git clone --recurse-submodules https://github.com/TallMessiWu/qwen3.8.git
cd qwen3.8
```

> 主仓地址按实际替换。`vllm` 指向上游 `vllm-project/vllm`，历史很大，完整克隆需要几分钟到十几分钟——想省时间看下面的浅克隆方案。

### 已经 clone 过、但 submodule 是空目录

克隆时忘了 `--recurse-submodules`，补一条即可：

```bash
git submodule update --init --recursive
```

### 省时方案：只对 vllm 浅克隆

`vllm/` 只用来查源码，不需要历史；`vllm-ascend/` 要开分支、做 worktree、跟 upstream 合并，**必须完整克隆**。所以分开拉：

```bash
git clone https://github.com/TallMessiWu/qwen3.8.git
cd qwen3.8
git submodule update --init --depth 1 vllm        # 上游参考，浅克隆
git submodule update --init vllm-ascend/main      # 开发主仓，完整历史
```

### 克隆后必做的两件事

**1. 给 vllm-ascend 补上 `upstream` remote。** `.gitmodules` 只记 `origin`（私仓），upstream 需要手动加：

```bash
cd vllm-ascend/main
git remote add upstream https://github.com/vllm-project/vllm-ascend.git
git fetch upstream
```

> ⚠️ **upstream 只 fetch，永远不 push。** 推送一律走 `origin`（私仓）。

**2. 把 vllm-ascend 从游离头指针切回 main 分支。** submodule 默认检出到 detached HEAD，这样没法提交：

```bash
cd vllm-ascend/main
git checkout main
git branch --set-upstream-to=origin/main main   # 若尚未关联
```

`vllm/` 保持 detached HEAD 就好——它是只读参考，不要修改、不要提交、不要推送。

### Windows 额外注意：符号链接

`CLAUDE.md` 和 `.claude/skills` 是符号链接。Windows 上如果没开 symlink 支持，clone 出来它们会变成装着路径字符串的普通文本文件，Claude Code 就读不到规则和技能了。

先确认：

```bash
git config --get core.symlinks     # 期望输出 true
```

不是 `true` 的话，开启后重新 clone：

```bash
git config --global core.symlinks true
git clone -c core.symlinks=true --recurse-submodules https://github.com/TallMessiWu/qwen3.8.git
```

还需要 Windows 允许创建符号链接——**打开「设置 → 系统 → 开发者选项 → 开发人员模式」**，或者用管理员权限的终端执行 clone。

### 验证克隆结果

```bash
git submodule status        # 两行，均无 - 前缀（有 - 说明未初始化）
cat CLAUDE.md | head -3     # 应输出 AGENTS.md 的正文，而不是 "AGENTS.md" 一行
ls .claude/skills/          # 应列出技能目录
```

## 日常同步

```bash
git pull                                  # 拉主仓（含 submodule 指针变化）
git submodule update --init --recursive   # 让子仓跟上新指针
```

submodule 指针只有在需要固定「这套脚本对应哪个版本的 vllm / vllm-ascend」时才更新，日常在子仓里提交**不必**顺手 bump：

```bash
git submodule status                      # 看两个子仓当前指向的 commit
git add vllm-ascend/main && git commit     # 需要时才推进指针
```

同步 vllm-ascend 上游（在 `vllm-ascend/main` 里做，再 rebase/merge 到特性分支）：

```bash
cd vllm-ascend/main
git fetch upstream && git merge upstream/main
```

## Worktree 工作流

每个任务一个分支一个目录，并行开发互不干扰。主 worktree 在 `vllm-ascend/main`，从那里派生：

```bash
cd vllm-ascend/main
git worktree add ../feat-xxx -b feat/xxx origin/main   # 新分支
git worktree add ../bugfix-yyy origin/bugfix/yyy       # 检出已有远程分支
git worktree list
git worktree remove ../feat-xxx                        # 收尾清理
```

目录名用分支名去掉斜杠（`feat/mxfp8-quant-group-tp` → `feat-mxfp8-quant-group-tp`）。分支命名沿用现有习惯：`feat/*`、`fix/*`、`bugfix/*`。

## 常用命令

在某个 vllm-ascend worktree 目录内执行：

```bash
# Lint / 格式化：提交前必跑，覆盖所有文件类型（含 markdown）
bash format.sh                           # 等价 pre-commit run --all-files
bash format.sh ci                        # CI 口径，含 manual stage 的钩子
pre-commit run ruff-check --all-files    # 只跑单个钩子

# 单元测试
pytest -sv tests/ut/ops/test_prepare_finalize.py

# e2e（需要 NPU 硬件，只能在服务器上跑）
pytest -sv tests/e2e/pull_request/one_card/aclgraph/test_aclgraph_accuracy.py
```

`tests/ut` 里大量用例 `import torch_npu`，本机跑不了；本机改动靠 lint + 静态阅读把关，真实验证交给服务器。

## 动手前先确认的硬性约束

1. **`vllm/` 是只读参考。** 需要改上游行为时，在 `vllm-ascend` 里写 patch。
2. **本机没有 NPU，跑不了推理。** 有一块 5080 GPU，可以跑纯 PyTorch 小脚本做数值等价性验证（不能 `import torch_npu`）。
3. **验证闭环靠脚本。** 可复现脚本写进 `scripts/`，自带打印/断言，输出能直接贴回来判读，不依赖交互式输入。
4. **只能推私仓。** `upstream` 只 fetch，永远不 push。
5. **不要主动 push。** 提交后停下，由人决定推送时机。
6. **提交走 `/gitmoji-commit` 技能。** 中文 subject，`<emoji-code> <type>(<scope>): <subject>` 格式。vllm-ascend 的 pre-commit 装了 `signoff-commit` 钩子，提交必须带 `-s`：

   ```bash
   git commit -s -m ":bug: fix(gdn): 修复 TP8 下 cumsum 分块导致的乱码"
   ```

完整规则（patch 分阶段机制、改动优先级、设备差异抽象、环境变量约定、NPU 性能红线、当前工作主线）见 [`AGENTS.md`](AGENTS.md)；vllm-ascend 仓内规范见 `vllm-ascend/main/AGENTS.md`。
