#!/usr/bin/env node
/**
 * 单文件架构页的静态体检：语法 / 字号 / 死代码。
 *
 *   node check.mjs <page.html> [--json]
 *
 * 三项检查都能在浏览器之外完成，改完随手跑一次，比开浏览器快得多。
 * 退出码 1 表示发现了必须修的问题（目前只有 JS 语法错误算硬失败，
 * 其余是提示——字号偏小和死代码都需要人判断，不该阻断流程）。
 */
import fs from 'node:fs';

const args = process.argv.slice(2);
const asJson = args.includes('--json');
const file = args.find(a => !a.startsWith('--'));

if (!file) {
  console.error('用法: node check.mjs <page.html> [--json]');
  process.exit(2);
}

const html = fs.readFileSync(file, 'utf8');
const report = { file, jsSyntax: null, fontSizes: null, deadCss: null };

/* ---------- 1. 内联 JS 语法 ---------- */
// new Function() 只解析不执行，所以不会因为缺少 DOM 而误报。
const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/);
if (!scriptMatch) {
  report.jsSyntax = { ok: false, error: '未找到内联 <script> 块' };
} else {
  try {
    new Function(scriptMatch[1]);
    report.jsSyntax = { ok: true, chars: scriptMatch[1].length };
  } catch (e) {
    report.jsSyntax = { ok: false, error: e.message };
  }
}

/* ---------- 2. 字号分布 ---------- */
// 这类信息密集的页面很容易一路写到 6~8px，单看某一处不觉得小，
// 汇总起来才会发现整页都在可读线以下。11px 是这里用的经验分界。
const styleMatch = html.match(/<style>([\s\S]*?)<\/style>/);
const css = styleMatch ? styleMatch[1] : '';
{
  const counts = {};
  const bump = n => { counts[n] = (counts[n] || 0) + 1; };
  let m;
  const reLong = /font-size:\s*([\d.]+)px/g;
  while ((m = reLong.exec(css))) bump(m[1]);
  // font 简写里第一个 px 值就是字号（line-height 用 /1.4 这种无单位写法）
  const reShort = /font:\s*[^;}"'`]*?([\d.]+)px/g;
  while ((m = reShort.exec(css))) bump(m[1]);

  const sizes = Object.keys(counts).map(Number).sort((a, b) => a - b);
  const total = sizes.reduce((s, k) => s + counts[k], 0);
  const under11 = sizes.filter(k => k < 11).reduce((s, k) => s + counts[k], 0);
  report.fontSizes = {
    total,
    under11,
    pctUnder11: total ? Math.round((under11 / total) * 100) : 0,
    distribution: sizes.map(k => ({ px: k, count: counts[k] })),
  };
}

/* ---------- 3. 死代码 CSS 类 ---------- */
// 只看 </style> 之后的正文，因为类名要么写在标签的 class 里，要么由 JS 拼出来。
// 动态拼接的类（`op-${accent}`、`tensor-${kind}`）扫不出来，必须显式豁免，
// 否则会把还在用的样式当成垃圾删掉。
const DYNAMIC_ALLOWLIST = [
  /^op-/,        // opStep(n,'gdn',…) → .op-gdn
  /^tensor-/,    // tensorPanel({kind:'after'}) → .tensor-after
  /^payload-/,   // 载荷按 kind 切换
  /^mono$/, /^sr$/, // 通用工具类
];
if (css) {
  const defined = new Set();
  let m;
  const reClass = /\.([a-zA-Z][a-zA-Z0-9_-]*)/g;
  while ((m = reClass.exec(css))) defined.add(m[1]);

  const body = html.slice(styleMatch.index + styleMatch[0].length);
  const dead = [...defined].filter(c => {
    if (DYNAMIC_ALLOWLIST.some(re => re.test(c))) return false;
    const re = new RegExp(`(^|[^a-zA-Z0-9_-])${c.replace(/-/g, '\\-')}([^a-zA-Z0-9_-]|$)`);
    return !re.test(body);
  }).sort();

  report.deadCss = { defined: defined.size, dead };
}

/* ---------- 输出 ---------- */
if (asJson) {
  console.log(JSON.stringify(report, null, 2));
} else {
  const js = report.jsSyntax;
  console.log(js.ok ? `JS 语法: OK (${js.chars} 字符)` : `JS 语法: 失败 — ${js.error}`);

  const f = report.fontSizes;
  console.log(`字号: 共 ${f.total} 处声明，${f.under11} 处 < 11px (${f.pctUnder11}%)`);
  if (f.pctUnder11 > 40) {
    console.log('  ⚠ 过半字号在可读线以下，考虑跑 scale-fonts.mjs 整体上调');
  }
  console.log('  ' + f.distribution.map(d => `${d.px}px×${d.count}`).join('  '));

  const d = report.deadCss;
  if (d) {
    console.log(`死代码 CSS: 定义 ${d.defined} 个类，${d.dead.length} 个未被引用`);
    if (d.dead.length) console.log('  ' + d.dead.join(', '));
    console.log('  （删之前先确认不是 JS 动态拼出来的类名）');
  }
}

process.exit(report.jsSyntax.ok ? 0 : 1);
