---
name: model-architecture-page
description: 为大模型制作可下钻的交互式架构网页（单文件 HTML）——总图展示请求到输出的完整路径，一份数据载荷在模块间连续移动并就地改变 shape 与数值，点击任意模块可下钻到内部逐步操作。当用户想把某个模型（Qwen、Llama、DeepSeek、GPT 等任意 Transformer）的架构做成可视化/交互页面、想讲清 Attention/GDN/MoE/FFN 内部到底做了哪几步、想对比多个模型的结构差异、或需要维护已有的架构图 HTML 时使用。用户提到"架构图""模型可视化""讲清楚模型内部""数据在模型里怎么变"时也应主动使用。
---

# 模型架构交互页

产出一个**单文件 HTML**（内联 CSS+JS、零外部依赖）：顶部切换模型，
总图按流程展开各模块，一份数据载荷沿路移动、每站就地变形，
点任意模块下钻到内部的逐步操作。

设计目标是**教学准确**：读者能看懂每一层做了哪几件事、数据的 shape 怎么变，
并且始终清楚哪些数字是官方配置、哪些是为了讲解编出来的。

## 先读哪一份

| 文件 | 何时读 |
|---|---|
| `references/design-principles.md` | 动手前通读。解释为什么这样画，以及每条规则对应的误读 |
| `references/component-api.md` | 填内容时查。数据结构、helper 函数、现成组件清单 |
| `references/layout-pitfalls.md` | 遇到布局怪象时查；改字号或改动画前必读 |

## 工作流

### 1. 先把架构事实核实清楚

**从官方 `config.json` 取数，不要凭印象写。** 这是唯一不能妥协的一步——
页面上的层数、hidden、头数一旦错了，整张图就是在传播错误。

```bash
curl -sL https://huggingface.co/<org>/<model>/resolve/main/config.json -o config.json
```

需要落实的字段：`hidden_size`、`num_hidden_layers`、`intermediate_size`、
`num_attention_heads`、`num_key_value_heads`、`head_dim`、`vocab_size`、
`tie_word_embeddings`、`max_position_embeddings`、`layer_types`（混合注意力的排布）、
MoE 相关（`num_experts` / `num_experts_per_tok` / `shared_expert`）、
以及 `vision_config`（多模态）。

如果模型尚未公开 config：**如实标注 TBD**，用 `tbd` 徽章把"已确认 / 合理先验 / 必须存疑"
分栏列出。绝不能拿同系列近似模型的数字顶替——那会变成一本正经的编造。

要对比两个模型是否同构，直接逐字段 diff 两份 config，别靠模型卡的描述。

### 2. 复制模板并填数据

```bash
cp .agents/skills/model-architecture-page/assets/skeleton.html <model>-architecture.html
```

模板开箱可跑（6 个主流程模块 + 2 个可下钻块）。改这四处即可换成目标模型：
`mainFlow` / `decoderFlow` / `moduleSpecs` / `payloadStates` + `meta`。
字段含义见 `references/component-api.md`。

`payloadStates` 的长度必须是 `moduleSpecs.length + 1`——最后一项是离开末模块后的形态。

### 3. 逐模块写"操作步骤"

这是页面的价值所在：**把每个块拆成它真正执行的几步操作**，而不是摆一堆张量。

```js
opStep(2,'gdn','短卷积','深度因果卷积 · 4-tap', null,
  'Q、K、V 各自沿时间做一次 depthwise 因果卷积：每个通道只看自己最近 4 个时刻。',
  opRow(tensorPanel({…}), opOp('Σ w·x + SiLU'), opScalars('教学标量',[['加权和','+.26']])))
```

**编号只发给真正的操作。** 结果展示、数值对照用 `opSub`（不编号）；
输入数据用 `opSource`（不编号）；并行分支用 `opParallel` 并列在同一编号内。

带圈数字在读者眼里就是"这一层做的第几件事"——把展示性内容也编号，
层的复杂度就被凭空放大了。这个坑真实发生过：RMSNorm 编了 ①重标定 ②数值对照，
读者第一反应是"原来 RMSNorm 有两个操作"。

某层确实只有一个操作时，在说明里直接点破，并说清它**不做**什么
（RMSNorm 没有均值中心化和偏置；Embedding 没有矩阵乘法）。

参考写法：GDN 拆 6 步（投影→短卷积→归一化门控→状态写入→读取→门控输出），
GQA 拆 6 步（投影→RoPE→KV缓存→打分→聚合→门控输出），
MoE 拆 5 步（路由→top-k→专家并行→共享专家→加权合并），
SwiGLU FFN 拆 3 步（上投影→门控→下投影）。

### 4. 跑验证

```bash
node .agents/skills/model-architecture-page/scripts/check.mjs <page>.html
```
检查内联 JS 语法、字号分布、未引用的 CSS 类。
（模板自带的组件库会被报成"未引用"，那是留给你用的，不是死代码。）

浏览器里粘贴 `scripts/audit-browser.js`，然后：

```js
await auditPage()        // 遍历全部模块：横向滚动条 / 元素溢出 / 文字裁切
await auditPage({widths:[1240,980,820]})   // 多宽度复测
await auditPayload()     // 载荷停靠是否遮挡卡片（每步等 1s，别采到飞行中间态）
await auditSticky()      // 控制台是否真的钉住
```

目标是 `pass: true`：滚动条 0、溢出 0、裁切 0。
`auditPayload` 期望 `overlappingDocks: 0` 且 `minGap > 0`（载荷在卡片下方）。

### 5. 修布局问题

审计报出来的问题，绝大多数能在 `references/layout-pitfalls.md` 里对号入座。
高频三条：

- **容器被撑破** → grid/flex item 加 `min-width:0`
- **出现横向滚动条** → 改 `flex-wrap:wrap`，不要 `overflow-x:auto`
- **sticky 失效** → 检查祖先的 `overflow-x:hidden`，换成 `clip`

字号整体偏小时：

```bash
node .agents/skills/model-architecture-page/scripts/scale-fonts.mjs <page>.html --also-sizes
```
按阶梯映射上调（小字提得多、大字提得少，保住层级差）。
**改完必然引出新的溢出点**，务必重跑第 4 步。

## 不能妥协的几条

这几条是页面可信度的底线，其余都可以按场景变通：

1. **架构数字来自官方 config**，没有就标 TBD，不拿近似模型顶替。
2. **所有编出来的数值常驻标注"模拟数据"**，张量面板同时给完整 shape 与切片坐标。
3. **编号只给真正的操作**（见上）。
4. **载荷不遮挡当前卡片**——停在卡片下方并用尖角指向它。
   读者最想看的恰恰是当前模块，把它盖住等于自毁。
5. **重复结构只画一组配方 + 乘数**（`×23 = 92 层`），
   FFN/MoE 画在每层内部，不作为独立一站。

## 维护已有页面

改动前先跑一遍第 4 步拿到基线，改完再跑一次对比。
动画和布局的耦合点集中在载荷定位、行末标记、sticky 三处，
改这些之前请先读 `layout-pitfalls.md` 对应小节。

参考实现：本仓根目录的 `qwen3.8-architecture.html`
（两个模型、13 个场景、含 GDN/GQA/MoE/Dense FFN 四类块的完整拆解）。
