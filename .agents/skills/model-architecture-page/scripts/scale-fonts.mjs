#!/usr/bin/env node
/**
 * 整页字号批量上调。
 *
 *   node scale-fonts.mjs <page.html> [--dry] [--also-sizes]
 *
 * 为什么需要它：信息密集的架构页写着写着就会整体偏小——单看某个标签
 * 觉得"再小一点能塞下"，累积起来整页都在可读线以下。逐条手改既慢又
 * 容易漏，而简单地全部乘以同一个系数又会把本来就大的标题顶到夸张。
 *
 * 这里用阶梯映射：小字提得多、大字提得少，放大后仍保留原有的层级差。
 *
 * --also-sizes 会一并上调与文字强相关的固定尺寸（行高、格宽、圆圈直径）。
 * 不加这个参数的话，字变大而容器没变，会直接把布局撑破。
 *
 * 改完务必跑 check.mjs + 浏览器 auditPage()，字号变化几乎一定会引出
 * 新的溢出点（见 references/layout-pitfalls.md 的 min-width:auto 一节）。
 */
import fs from 'node:fs';

const args = process.argv.slice(2);
const dry = args.includes('--dry');
const alsoSizes = args.includes('--also-sizes');
const file = args.find(a => !a.startsWith('--'));

if (!file) {
  console.error('用法: node scale-fonts.mjs <page.html> [--dry] [--also-sizes]');
  process.exit(2);
}

// 阶梯映射：11px 以下大幅提升（这些是最影响可读性的），越大提升越保守。
const FONT_MAP = {
  '5': 9, '5.5': 9, '6': 9.5, '6.5': 10, '7': 10.5, '7.5': 11,
  '8': 11.5, '8.5': 12, '9': 12.5, '9.5': 12.5, '10': 13, '10.5': 13.5,
  '11': 13.5, '12': 14.5, '12.5': 15, '13': 15.5, '14': 16.5, '15': 17.5,
  '16': 18.5, '18': 21, '20': 23, '25': 28, '27': 30, '28': 31, '29': 32, '31': 34,
};

// 与文字直接绑定的固定尺寸，字号涨了它们必须跟着涨。
// 这里只列通用的；模板特有的尺寸请按需补充。
const SIZE_MAP = [
  ['--header:56px', '--header:64px'],
  ['grid-template-rows:38px 26px', 'grid-template-rows:46px 32px'],   // 层配方色块
  ['width:23px;height:23px', 'width:28px;height:28px'],               // 步骤序号圆
  ['min-height:176px', 'min-height:212px'],                           // 总图卡片
];

let html = fs.readFileSync(file, 'utf8');
const styleMatch = html.match(/<style>([\s\S]*?)<\/style>/);
if (!styleMatch) { console.error('未找到 <style> 块'); process.exit(2); }

let css = styleMatch[1];
const changes = [];
const map = n => (FONT_MAP[n] !== undefined ? FONT_MAP[n] : n);

css = css.replace(/font-size:\s*([\d.]+)px/g, (m, n) => {
  const to = map(n);
  if (String(to) !== n) changes.push(`${n}→${to}`);
  return `font-size:${to}px`;
});
// font 简写：非贪婪匹配到第一个 px，且不跨越 ; } 和引号，避免误伤同规则的其它属性
css = css.replace(/(font:[^;}"'`]*?)([\d.]+)px/g, (m, pre, n) => {
  const to = map(n);
  if (String(to) !== n) changes.push(`${n}→${to}`);
  return `${pre}${to}px`;
});

let sizeHits = 0;
if (alsoSizes) {
  for (const [from, to] of SIZE_MAP) {
    if (css.includes(from)) { css = css.split(from).join(to); sizeHits++; }
  }
}

const tally = changes.reduce((acc, c) => { acc[c] = (acc[c] || 0) + 1; return acc; }, {});
console.log(`字号改写 ${changes.length} 处:`);
console.log('  ' + Object.entries(tally).map(([k, v]) => `${k}(×${v})`).join('  '));
if (alsoSizes) console.log(`固定尺寸同步 ${sizeHits}/${SIZE_MAP.length} 处`);

if (dry) {
  console.log('\n--dry 模式，未写入。');
} else {
  html = html.slice(0, styleMatch.index) + '<style>' + css + '</style>' +
         html.slice(styleMatch.index + styleMatch[0].length);
  fs.writeFileSync(file, html);
  console.log(`\n已写入 ${file}`);
  console.log('接着跑: node check.mjs ' + file + '  然后在浏览器 await auditPage()');
}
