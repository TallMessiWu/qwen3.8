# 布局陷阱清单

这些坑都是在真实页面上踩出来的，每一条都伴随过一次"看起来没道理"的调试。
按"症状 → 原因 → 修法"组织，方便你从现象反查。

## 目录

- [sticky 莫名失效](#sticky-莫名失效)
- [容器被内容撑破而不是收缩](#容器被内容撑破而不是收缩)
- [张量网格：语义列 vs 自动列](#张量网格语义列-vs-自动列)
- [横向滚动条 vs 换行](#横向滚动条-vs-换行)
- [scrollWidth 误报：绝对定位装饰件](#scrollwidth-误报绝对定位装饰件)
- [测量动画元素采到中间态](#测量动画元素采到中间态)
- [固定宽容器装不下放大后的文字](#固定宽容器装不下放大后的文字)
- [display:block 污染行内强调](#displayblock-污染行内强调)
- [载荷比卡片宽时顶出边界](#载荷比卡片宽时顶出边界)

---

## sticky 莫名失效

**症状**：`position:sticky` 的元素照常滚走，连顶栏也不钉住。CSS 看起来完全正确。

**原因**：祖先元素上的 `overflow-x:hidden`。它会把该元素变成滚动容器，
sticky 于是相对这个"不滚动的容器"定位，永远不会触发。
`body{overflow-x:hidden}` 是最常见的元凶——很多人拿它防横向滚动条。

同类问题还有一层：`.canvas{overflow:auto}` 这种为内部滚动准备的容器，
在窄屏布局下如果实际滚动的是 window，sticky 也会失效。

**修法**：
```css
/* 防横向溢出用 clip，它不创建滚动容器 */
body{overflow-x:clip}

/* 内部滚动容器只在真正需要内部滚动的断点保留 */
@media(max-width:1300px){ .canvas{overflow:visible} }
```

两种布局下 sticky 的参照物不同，`top` 也要跟着换：
容器内滚动时 `top:0` 即可；window 滚动时要让开固定顶栏，`top:var(--header)`。

**验证**：滚动到几个不同位置，看它的 `getBoundingClientRect().top` 是否恒定。
`scripts/audit-browser.js` 的 `auditSticky()` 就是干这个的。

---

## 容器被内容撑破而不是收缩

**症状**：某个 grid/flex 子项里放了宽内容，结果整个容器变宽并溢出，
而不是内容自己换行或压缩。

**原因**：grid item 和 flex item 的 `min-width` 默认是 `auto`，
意思是"不得小于内容的最小宽度"。内容不可断行时，容器就被顶大。

**修法**：给可能装下宽内容的容器显式声明 `min-width:0`。

```css
.op-pipeline{display:grid;min-width:0}
.op-step{min-width:0}      /* grid item */
.map-node{min-width:0}     /* grid item */
.tensor-panel{min-width:0;max-width:100%}
```

这条在字号放大后会集中爆发——原本刚好塞下的内容突然放不下了。

---

## 张量网格：语义列 vs 自动列

**症状**：想让放不下的数值换行，改用 `repeat(auto-fit,minmax(38px,1fr))`
之后，一个本该是 3 行 × 4 列的张量（row0=Q / row1=K / row2=V）
被拉成了 11 列 + 1 列。

**原因**：`auto-fit` 按容器宽度决定列数，而张量的列数是**语义**——
每行对应张量的一行。列数一变，行列关系就没了，读者会误读数据结构。

**修法**：保持语义列数，用 `minmax(0,1fr)` 让列严格均分容器。
这样宽时自然舒展、窄时压缩，且永远不会撑破。

```css
.tensor-grid{grid-template-columns:repeat(var(--tc),minmax(0,1fr))}
.tensor-grid span{min-width:0;overflow:hidden}
```

注意 `minmax(34px,1fr)` 这类写法仍会被长内容（如 6 位 token ID）撑大，
所以最小值要给 `0`，靠均分保证可读宽度。

**如果均分后仍然太挤**（例如并行分支只有半行宽），
不要牺牲列语义，改为在那个上下文里降一档字号：

```css
.op-par-lane .tensor-grid span{font-size:9.5px;padding:5px 1px}
```

---

## 横向滚动条 vs 换行

信息密集的页面里，横向滚动条几乎总是错的选择：读者不知道右边还有内容，
拖动也打断阅读节奏。放不下就换行。

```css
/* 不要 */
.op-body{display:flex;overflow-x:auto}

/* 要 */
.op-body{display:flex;flex-wrap:wrap;gap:10px 9px}
.op-body>*{flex:0 1 auto;min-width:0;max-width:100%}
```

换行后要复查两件事：单个子元素是否宽于容器（换行也救不了），
以及换行是否破坏了语义顺序（箭头落到行首会读不通）。

---

## scrollWidth 误报：绝对定位装饰件

**症状**：审计脚本报告某个容器溢出 20px，但逐个子元素查过去都没超。

**原因**：绝对定位且**故意**探出父容器的装饰件（如卡片右侧外的 `→` 箭头，
`right:-20px`）会计入父容器的 `scrollWidth`。

**修法**：这不是真问题——只要它落在容器间隙内、不与相邻元素重叠即可。
审计时把这类元素的容器加进豁免名单，否则一堆假阳性会淹没真问题。

真正该看的指标是**网格容器**和**页面**的横向溢出是否为 0。

---

## 测量动画元素采到中间态

**症状**：脚本报告"载荷与卡片重叠"，但肉眼看明明停得好好的。

**原因**：载荷的 `transition` 约 0.9s。步进后立刻测量，
拿到的是飞行途中的位置，不是停靠位置。

**修法**：每步等 transition 完整走完（1000ms）再测。
如果按 150ms 采样，会得到一串毫无意义的"重叠"结论。

```js
step.click();
await sleep(1000);          // 关键
const rect = payload.getBoundingClientRect();
```

判断停靠是否正确，看 `payload.top - activeCard.bottom` 是否为正
（正数 = 载荷在卡片下方）。

---

## 固定宽容器装不下放大后的文字

字号一改，所有写死的宽高都可能不够。典型受害者：

| 位置 | 症状 |
|---|---|
| 标签列 `grid-template-columns:30px 1fr` | 两字标签被挤成两行 |
| 柱状图容器固定 210px，内含 28 根 `min-width:4px` 的柱子 | 柱子溢出并压住旁边的文字 |
| 分布行第三列 42px，内容 `13.40→0.289` | 撑破整行 |
| 顶栏按钮 6 个 | 总宽超出，面包屑被压成 0 宽后竖排 |

**修法**：字号调整必须连带调整这些尺寸（`scripts/scale-fonts.mjs --also-sizes`
覆盖了通用的几处，其余要按报错逐个补）。顶栏这类控件密集区可以保持比正文小一档，
不必和正文同步放大。

---

## display:block 污染行内强调

**症状**：段落里的 `<b>` 强调把句子拆成了好几行。

**原因**：为标题写的 `.dec-copy b{display:block}` 会命中所有后代 `<b>`，
包括段落里当作行内强调用的那些。

**修法**：用直接子代选择器把标题和行内强调分开。

```css
.dec-copy>b{display:block;font:800 15px var(--display)}  /* 标题 */
.dec-copy p b{color:var(--ink);font-weight:800}          /* 行内强调 */
```

---

## 载荷比卡片宽时顶出边界

**症状**：载荷停在最左或最右列时，把总图容器顶出十几像素。

**原因**：载荷宽度固定（如 304px），可能略宽于卡片列宽；
它以卡片中心为锚点居中，边列自然会探出容器。

**修法**：把 x 坐标钳制在容器内，同时让尖角**单独**对准卡片中心，
这样即使载荷被推回边界内，指向依然准确。

```js
const anchorX = portRect.left - mapRect.left + portRect.width/2;
const x = Math.min(Math.max(pad, anchorX - pw/2),
                   Math.max(pad, map.clientWidth - pw - pad));
return { x, y, arrow: anchorX - x };   // arrow 写进 --arrow-x
```

```css
.travelling-payload::before{left:var(--arrow-x,50%);transition:left .88s}
```
