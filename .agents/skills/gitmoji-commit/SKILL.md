---
name: gitmoji-commit
description: 根据当前暂存区的代码变更和会话上下文，生成符合 Gitmoji 规范的中文提交信息并执行提交（不推送）。
---

# Gitmoji Commit Skill

此 Skill 用于帮助用户自动化 Git 提交流程。你需要分析代码变更，结合当前的会话上下文（用户之前的指令和你的修改），生成符合 [Gitmoji](https://gitmoji.dev/) 规范的提交信息。

## 核心原则
1.  **始终使用中文**撰写 Subject。
2.  **严禁推送 (git push)**，仅执行本地提交 (git commit)。
3.  **格式规范**: `<emoji-code> <type>(<scope>): <subject>`
    *   **Emoji**: **必须**使用 Gitmoji 代码（如 `:sparkles:`）以便兼容性。
    *   **Subject**: 简练的中文描述，动词开头，不超过 50 个字符。作为第一个 `-m` 的标题。
    *   **Body (可选)**: **强烈建议不要使用两个 `-m` 参数**，除非 commit 包含大量不同层面的复杂修改，且仅看标题无法清晰传达详细信息时才使用。一般情况下，只要 `Subject` 能够概括核心意图，就应尽量保持语言简洁，仅使用一个 `-m` 参数即可。
    *   Example (默认推荐 - 简洁): `git commit -m ":sparkles: feat(auth): 添加登录功能"`
    *   Example (仅限复杂变更 - 拆分细节): `git commit -m ":bug: fix(nav): 修复导航栏样式偏移" -m "1. 修复了移动端下 margin 计算错误的 bug\n2. 统一了背景毛玻璃组件的 z-index"`

## 执行步骤

### 1. 检查暂存区
首先，检查是否有已暂存的文件：
```powershell
git diff --cached --name-only
```
*   **若无暂存文件**: 提示用户“检测到暂存区为空，请先使用 `git add` 暂存文件”，并停止执行。
*   **若有暂存文件**: 继续下一步。

### 2. 分析变更与上下文
运行 `git diff --cached` 获取详细变更内容。同时，**必须**回顾当前的会话历史：
-   用户刚才要求做什么？
-   代码的具体变动是什么？
-   如果有多个变更，**主要意图**是什么？（例如：如果既修改了构建脚本又修复了 bug，优先选择影响最大的那个，或者分两次提交——但在本 Skill 中，默认生成一个包含主要变更意图的提交）。

### 3. 生成提交信息
根据分析结果，从下表中选择**最准确**的 Emoji 和 Type。

#### Gitmoji 参考手册

| Emoji | Code | Description | 中文含义（参考） |
| :--- | :--- | :--- | :--- |
| 🎨 | `:art:` | Improve structure / format of the code. | 改进代码结构/格式 |
| ⚡️ | `:zap:` | Improve performance. | 性能优化 |
| 🔥 | `:fire:` | Remove code or files. | 删除代码/文件 |
| 🐛 | `:bug:` | Fix a bug. | 修复 Bug |
| 🚑️ | `:ambulance:` | Critical hotfix. | 紧急热修复 |
| ✨ | `:sparkles:` | Introduce new features. | 引入新功能 |
| � | `:memo:` | Add or update documentation. | 添加/更新文档 |
| 🚀 | `:rocket:` | Deploy stuff. | 部署/发布 |
| 💄 | `:lipstick:` | Add or update the UI and style files. | 更新 UI/样式 |
| 🎉 | `:tada:` | Begin a project. | 初次提交/初始化项目 |
| ✅ | `:white_check_mark:` | Add, update, or pass tests. | 添加/更新/通过测试 |
| 🔒 | `:lock:` | Fix security or privacy issues. | 修复安全/隐私问题 |
| 🔐 | `:closed_lock_with_key:` | Add or update secrets. | 添加/更新密钥 |
| 🔖 | `:bookmark:` | Release / Version tags. | 发布/版本标签 |
| 🚨 | `:rotating_light:` | Fix compiler / linter warnings. | 修复编译器/Linter 警告 |
| 🚧 | `:construction:` | Work in progress. | 进行中的工作 |
| 💚 | `:green_heart:` | Fix CI Build. | 修复 CI 构建 |
| ⬇️ | `:arrow_down:` | Downgrade dependencies. | 降级依赖 |
| ⬆️ | `:arrow_up:` | Upgrade dependencies. | 升级依赖 |
| 📌 | `:pushpin:` | Pin dependencies to specific versions. | 锁定依赖版本 |
| 👷 | `:construction_worker:` | Add or update CI build system. | 添加/更新 CI 构建系统 |
| 📈 | `:chart_with_upwards_trend:` | Add or update analytics or track code. | 添加/更新分析或追踪代码 |
| ♻️ | `:recycle:` | Refactor code. | 代码重构 |
| ➕ | `:heavy_plus_sign:` | Add a dependency. | 添加依赖 |
| ➖ | `:heavy_minus_sign:` | Remove a dependency. | 移除依赖 |
| 🔧 | `:wrench:` | Add or update configuration files. | 添加/更新配置文件 |
| � | `:hammer:` | Add or update development scripts. | 添加/更新开发脚本 |
| 🌐 | `:globe_with_meridians:` | Internationalization and localization. | 国际化与本地化 |
| ✏️ | `:pencil2:` | Fix typos. | 修复拼写错误 |
| 💩 | `:poop:` | Write bad code that needs to be improved. | 提交需要改进的劣质代码 |
| ⏪ | `:rewind:` | Revert changes. | 回滚变更 |
| 🔀 | `:twisted_rightwards_arrows:` | Merge branches. | 合并分支 |
| 📦️ | `:package:` | Add or update compiled files or packages. | 更新编译文件或包 |
| 👽️ | `:alien:` | Update code due to external API changes. | 因外部 API 变更更新代码 |
| 🚚 | `:truck:` | Move or rename resources (e.g.: files, paths, routes). | 移动/重命名资源 |
| 📄 | `:page_facing_up:` | Add or update license. | 添加/更新许可证 |
| � | `:boom:` | Introduce breaking changes. | 引入破坏性变更 |
| 🍱 | `:bento:` | Add or update assets. | 添加/更新资源(图片等) |
| ♿️ | `:wheelchair:` | Improve accessibility. | 改进无障碍访问 |
| 💡 | `:bulb:` | Add or update comments in source code. | 添加/更新注释 |
| 🍻 | `:beers:` | Write code drunkenly. | 醉酒写代码(通常用于非正式提交) |
| 💬 | `:speech_balloon:` | Add or update text and literals. | 更新文本/字面量 |
| 🗃️ | `:card_file_box:` | Perform database related changes. | 数据库相关变更 |
| � | `:loud_sound:` | Add or update logs. | 添加/更新日志 |
| 🔇 | `:mute:` | Remove logs. | 移除日志 |
| 👥 | `:busts_in_silhouette:` | Add or update contributor(s). | 添加/更新贡献者 |
| 🚸 | `:children_crossing:` | Improve user experience / usability. | 改进用户体验/可用性 |
| 🏗️ | `:building_construction:` | Make architectural changes. | 架构变更 |
| 📱 | `:iphone:` | Work on responsive design. | 响应式设计 |
| 🤡 | `:clown_face:` | Mock things. | 模拟数据/Mock |
| 🥚 | `:egg:` | Add or update an easter egg. | 添加/更新彩蛋 |
| � | `:see_no_evil:` | Add or update a .gitignore file. | 添加/更新 .gitignore |
| 📸 | `:camera_flash:` | Add or update snapshots. | 添加/更新快照 |
| ⚗️ | `:alembic:` | Perform experiments. | 试验性代码 |
| 🔍 | `:mag:` | Improve SEO. | 改进 SEO |
| 🏷️ | `:label:` | Add or update types. | 添加/更新类型(TypeScript等) |
| 🌱 | `:seedling:` | Add or update seed files. | 添加/更新种子文件 |
| � | `:triangular_flag_on_post:` | Add, update, or remove feature flags. | 功能标记变更 |
| 🥅 | `:goal_net:` | Catch errors. | 错误捕获 |
| 💫 | `:dizzy:` | Add or update animations and transitions. | 动画/过渡效果 |
| 🗑️ | `:wastebasket:` | Deprecate code that needs to be cleaned up. | 废弃代码 |
| 🛂 | `:passport_control:` | Work on code related to authorization, roles and permissions. | 授权/角色/权限相关 |
| 🩹 | `:adhesive_bandage:` | Simple fix for a non-critical issue. | 简单修复非关键问题 |
| 🧐 | `:monocle_face:` | Data exploration/inspection. | 数据探索/检查 |
| ⚰️ | `:coffin:` | Remove dead code. | 移除死代码 |
| 🧪 | `:test_tube:` | Add a failing test. | 添加失败测试 |
| 👔 | `:necktie:` | Add or update business logic. | 业务逻辑变更 |
| 🩺 | `:stethoscope:` | Add or update healthcheck. | 健康检查 |
| 🧱 | `:bricks:` | Infrastructure related changes. | 基础设施变更 |
| 🧑‍� | `:technologist:` | Improve developer experience. | 改进开发者体验 |
| 💸 | `:money_with_wings:` | Add sponsorships or money related infrastructure. | 赞助/资金相关 |
| 🧵 | `:thread:` | Add or update code related to multithreading or concurrency. | 多线程/并发相关 |
| 🦺 | `:safety_vest:` | Add or update code related to validation. | 验证相关代码 |
| ✈️ | `:airplane:` | Improve offline support. | 离线支持 |
| 🦖 | `:t-rex:` | Code that adds backwards compatibility. | 向后兼容性 |
| 🤖 | `:robot:` | Changes related to AI agents, Codex config, or automation scripts. | Agent/AI 配置变更 |

*(Agent 请注意：选择时请优先匹配最具体的情境。例如：如果是更新 `package.json` 的版本号，用 `:arrow_up:` 或 `:arrow_down:` 或 `:heavy_plus_sign:` 比通用的 `:package:` 或 `:wrench:` 更好。如果是单纯的样式修改，必须用 `:lipstick:`)*

### 4. 用户交互与执行
**必须**先向用户展示你生成的提交命令，并请求确认。

**示例对话**:
> **Agent**: 暂存区包含 `package.json` 的依赖更新。
> 建议提交信息：
> `git commit -m ":arrow_up: chore(deps): 升级 vue 版本至 3.4"`
>
> 是否执行？

**只有在用户明确回复“是”、“确认”或“ok”后**，才执行 `run_command`：
```powershell
# 简单 Commit
git commit -m ":your-emoji: type(scope): subject"

# 复杂 Commit
git commit -m ":your-emoji: type(scope): subject" -m "1. 详细变更一\n2. 详细变更二"
```

## 注意事项
-   如果用户对生成的提交信息不满意，请根据用户的反馈进行调整，再次请求确认。
-   提交成功后，告知用户提交已完成，并提醒用户自行推送 (push)。
