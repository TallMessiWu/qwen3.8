# 组件与 API 参考

模板 `assets/skeleton.html` 里已内置的构件。按"数据 → 步骤 → 展示"分组。

## 目录

- [数据结构](#数据结构)
- [步骤构件](#步骤构件)
- [布局构件](#布局构件)
- [张量与数值](#张量与数值)
- [现成可视化组件](#现成可视化组件)
- [配色与徽章](#配色与徽章)

---

## 数据结构

换模型时主要改这四处：

```js
const mainFlow   = ['request','tokenizer','embedding','decoder','lmhead','output'];
const decoderFlow= ['attn','ffn'];        // Decoder 内可下钻的块
const shortLabel = {request:'请求', …};   // 上/下一模块按钮上的短名

const moduleSpecs = [
  { key:'embedding',            // 唯一标识，同时是 data-open 的值
    label:'Embedding',          // 卡片标题
    sub:'整数索引查表，得到连续向量',  // 卡片上的一句话说明
    shape:'[B,T] → [B,T,H]',    // 形态变化
    op:'E[input_ids]' }         // 这一站做什么操作
];

// 载荷在每一站的形态；长度必须 = moduleSpecs.length + 1
// （最后一项是离开末模块后的形态，ledger 的"离开"列要用）
const payloadStates = [
  { title:'input_ids', kind:'ids', shape:'[B,T] int64', values:[391,72,114], cols:3 }
];

const meta = {           // 右侧检查器的内容，键与 moduleSpecs 的 key 对应
  embedding:{ title, scope, copy, io:[[标签,值]], formula, badges:[], evidence }
};
```

`payloadStates[].kind` 决定载荷的渲染方式：

| kind | 渲染 | 需要的字段 |
|---|---|---|
| `json` | 深色 JSON 块 | `text` |
| `text` | 紫色单行文本 | `text` |
| `ids` | 整数网格 | `values`, `cols` |
| `tensor` | 浮点网格（带正负配色） | `values`, `cols` |
| `logits` | 同 tensor，按 14 归一化配色 | `values`, `cols` |
| `prob` | 同 tensor，按 0.3 归一化，保留 3 位小数 | `values`, `cols` |
| `result` | 绿色结果块 | `text` |

---

## 步骤构件

**编号只发给真正的操作**——原因见 `design-principles.md`。

```js
opStep(n, accent, kind, title, shape, desc, body)
```
| 参数 | 说明 |
|---|---|
| `n` | 编号，从 1 开始 |
| `accent` | 配色：`main`(蓝) / `gdn`(青) / `gqa`(紫) / `moe`(橙) / `ffn`(绿) |
| `kind` | 左侧小标签，如 `投影` `打分` |
| `title` | 操作名 |
| `shape` | 形态变化，可传 `null` |
| `desc` | 一句话讲清这步在干什么，支持 `<code>` `<b>` |
| `body` | 用下面的布局构件拼 |

```js
opPipe([step1, step2, …])   // 包住所有步骤，画出连接线
```

**非操作元素**（都不编号）：

```js
opSource(kind, title, shape, desc, body)   // 输入数据：虚线 IN 块
opSub(title, body, tag='示例数值')          // 结果展示/数值对照：步骤内灰色次级区
opParallel(note, [[标题, 说明, 内容], …])    // 并行分支：同一编号内并列
```

`opSub` 放在 `opStep` 的 `body` 里（字符串拼接），表示"这是该操作的附属展示"：

```js
opStep(1,'main','重标定','RMSNorm','[B,T,H] → 同 shape',
  '<b>这一层只有一个操作。</b>…',
  opRow(…) + opSub('同一操作的数值对照', opRow(…)))
```

---

## 布局构件

```js
opRow(...parts)        // 一行，自动换行（不产生横向滚动条）
opBlock(inner)         // 整块，用于不需要横排的内容
opArrow('→')           // 简单箭头
opOp(label,'→')        // 箭头 + 下方小标签，用于标注这一步的算子
opSign('⊙')            // 数学符号（⊙ Δ = 等）
opScalars('教学标量',[['β','.73'],['decay','.94']])   // 一排标量
opWidget(width, inner) // 固定宽度的槽位，放自定义可视化
```

---

## 张量与数值

```js
tensorPanel({
  name:'Sₜ 写入后',
  shape:'[B,128,128,128]',      // 完整逻辑 shape
  slice:'[0,head 0,0:4,0:4]',   // 当前切片坐标
  values:[…], cols:4,
  kind:'after',                 // ''(默认) / 'after'(蓝) / 'delta'(金)
  axis:'列 = t−3, t−2, t−1, t', // 可选，坐标说明
  format:'float'                // 'float' 或 'int'
})
```

面板会常驻显示"模拟数据"角标，并同时给出完整 shape 与切片坐标——
读者必须知道自己在看整个张量的哪一块。

数值生成用确定性伪随机，保证每次刷新数值一致：

```js
vals(n, seed)              // n 个 [-1,1] 的确定性数值
vector(n, seed)            // 竖条带
matrix(rows, cols, seed)   // 热力图方阵
bars(n, seed, color)       // 柱状条
round(v, d=2)
```

---

## 现成可视化组件

这些 CSS 类模板里已带，直接写 HTML 即可用（骨架示例未全部用到，
所以 `check.mjs` 会把它们报成"未引用"——那是组件库，不是死代码）：

| 类 | 用途 |
|---|---|
| `.json-box` + `.key` `.val` | 深色 JSON 展示 |
| `.template-lines` / `.template-line` | chat template 逐行 |
| `.token-bench` / `.token-piece`(`.special`) | token 切分展示 |
| `.embed-matrix` + `.row`(`.target`) | 嵌入矩阵取行 |
| `.vector-strip` / `.matrix` | 向量条带 / 热力图 |
| `.bar-vector` | 归一化前后的柱状对比 |
| `.branch-box` / `.branch` | 投影 fan-out 分支 |
| `.qkv-stack` | Q/K/V 堆叠标签 |
| `.rope-disc` | RoPE 旋转盘 |
| `.kv-cache` | KV 缓存格 |
| `.score-bars` | 注意力分数条 |
| `.router-field` | MoE 路由打分条 |
| `.experts` / `.expert` / `.shared-expert` | 专家网格 |
| `.logits` / `.logit`(`.selected`) | logits 排行 |
| `.distribution` / `.dist-row`(`.cut` `.pick`) | 采样分布 |
| `.sample-pointer` | 采样落点 |
| `.output-tokens` / `.out-token` / `.sentence-out` | 输出拼句 |
| `.stream-lines` + `.reason` `.answer` | 流式事件 |
| `.vit-stack` / `.patch-grid` / `.image-card` | 视觉塔 |
| `.macro-progress` | 层块进度条 |
| `.data-node`(`.active-border`) | 通用数据节点 |

---

## 配色与徽章

CSS 变量（`:root`）：

| 变量 | 语义 |
|---|---|
| `--green` / `--green2` | 文本、token、Dense FFN、服务边界 |
| `--cyan` / `--cyan2` | 线性注意力 / GDN |
| `--purple` / `--purple2` | 全注意力 / GQA |
| `--orange` / `--orange2` | MoE 路由、专家、采样 |
| `--gold` / `--gold2` | 模拟数据标注 |
| `--blue` / `--blue2` | 残差流、确定 shape、主流程 |

徽章 `badge(type)`：

| type | 显示 | 何时用 |
|---|---|---|
| `fact` | 配置事实 | 来自官方 config.json |
| `impl` | 参考实现 | 来自 transformers 等参考代码 |
| `engine` | 引擎相关 | 取决于推理引擎（vLLM 等） |
| `demo` | 模拟数据 | 页面上编出来的数值 |
| `tbd` | 尚未公开 | 官方未发布，保持未知 |
