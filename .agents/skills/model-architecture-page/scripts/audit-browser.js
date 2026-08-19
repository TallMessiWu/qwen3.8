/**
 * 架构页的运行时布局审计。整段粘进浏览器控制台执行，或用自动化工具求值。
 *
 *   await auditPage()                       // 全部模型 × 全部模块
 *   await auditPage({widths:[1240,820]})    // 额外在指定画布宽度下复测
 *   await auditPayload()                    // 单独查载荷停靠是否遮挡卡片
 *
 * 返回结构化结果，空数组/零计数即通过。
 *
 * 之所以要专门写这个而不是靠肉眼截图：这类页面模块多、状态多，
 * 一处溢出往往只在某个模块的某个宽度下出现，翻截图翻不全也记不住。
 */

/* ============ 通用工具 ============ */

// 绝对定位的装饰件（卡片外侧的 → 箭头）本来就该探出父容器，
// 它会被算进 scrollWidth，不排除就会得到一堆假阳性。
const DECORATIVE = ['map-node'];

function firstClass(el) {
  return (el.className || '').toString().split(' ')[0] || el.tagName.toLowerCase();
}

function scenesOf(model) {
  // 页面自己知道有哪些模块，直接问它，避免在脚本里硬编码模块名
  const sel = document.getElementById('modelSelect');
  if (sel) { sel.value = model; sel.dispatchEvent(new Event('change', { bubbles: true })); }
  const specs = window.__MAP_SPECS__;              // 模板会挂出来；没有就退回 DOM 探测
  if (Array.isArray(specs)) return specs;
  return [...document.querySelectorAll('.map-node[data-open]')].map(n => n.dataset.open);
}

function openScene(key) {
  const back = document.getElementById('overviewBtn');
  if (back && !back.hidden) back.click();
  if (key === 'overview') return;
  let el = document.querySelector(`[data-open="${key}"]`);
  if (el) { el.click(); return; }
  // Decoder 内部的块要先进 Decoder 才点得到
  const dec = document.querySelector('[data-open="decoder"]');
  if (dec) dec.click();
  el = document.querySelector(`[data-open="${key}"]`);
  if (el) el.click();
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

/* ============ 主审计 ============ */

async function auditPage(opts = {}) {
  const models = opts.models || [...document.querySelectorAll('#modelSelect option')].map(o => o.value);
  const widths = opts.widths || [null];            // null = 当前自然宽度
  const host = document.querySelector('.canvas-inner');
  const out = { scrollbars: [], overflow: [], clipped: [], byWidth: {} };

  for (const w of widths) {
    if (w && host) host.style.maxWidth = w + 'px';
    const tag = w ? `@${w}` : '@natural';
    const bucket = { scrollbars: 0, overflow: 0, clipped: 0 };

    for (const model of models) {
      for (const key of scenesOf(model)) {
        openScene(key);
        await sleep(30);                            // 让 render 落地

        const scope = document.querySelector('.canvas');
        for (const el of scope.querySelectorAll('*')) {
          const cs = getComputedStyle(el);
          const canScroll = cs.overflowX === 'auto' || cs.overflowX === 'scroll';
          const over = el.scrollWidth - el.clientWidth;
          if (over <= 3 || el.clientWidth === 0) continue;
          const cls = firstClass(el);

          if (canScroll) {
            // 出现了真实横向滚动条 —— 这类页面应该靠换行而不是拖动
            bucket.scrollbars++;
            out.scrollbars.push(`${tag} ${model}/${key} .${cls} +${over}`);
          } else if (!DECORATIVE.includes(cls)) {
            bucket.overflow++;
            out.overflow.push(`${tag} ${model}/${key} .${cls} +${over} (w=${el.clientWidth})`);
          }
        }

        // 文本被裁：数值/标签看不全，等于信息丢失
        const textish = '.tensor-grid span,.tensor-cell,.node-line b,.node-line code,' +
                        '.mixer-chip,.moe-chip,.ffn-chip,.tensor-name,.op-kind';
        for (const el of document.querySelectorAll(textish)) {
          if (el.scrollWidth - el.clientWidth > 1) {
            bucket.clipped++;
            out.clipped.push(`${tag} ${model}/${key} '${(el.textContent || '').trim().slice(0, 20)}'`);
          }
        }
      }
    }
    out.byWidth[tag] = bucket;
  }

  if (host) host.style.maxWidth = '';
  out.pageOverflowX = document.documentElement.scrollWidth - document.documentElement.clientWidth;
  out.pass = out.scrollbars.length === 0 && out.overflow.length === 0 &&
             out.clipped.length === 0 && out.pageOverflowX === 0;
  return out;
}

/* ============ 载荷停靠审计 ============ */

/**
 * 载荷必须停在卡片"下方"而不是"盖住卡片"。
 * 关键：CSS transition 约 0.9s，采样太早会拍到飞行中间态，
 * 得到一堆假的"重叠"结论。这里每步固定等 1s。
 */
async function auditPayload(steps = 12) {
  const toggle = document.getElementById('traceToggleBtn');
  if (toggle && toggle.textContent.includes('暂停')) toggle.click();  // 先暂停自动播放
  const step = document.getElementById('traceStepBtn');
  const payload = document.getElementById('travellingPayload');
  const map = document.getElementById('overviewMap');
  if (!step || !payload || !map) return { error: '未找到载荷或步进控件' };

  const samples = [];
  for (let i = 0; i < steps; i++) {
    step.click();
    await sleep(1000);                              // 等 transition 完全走完

    const pr = payload.getBoundingClientRect();
    const cards = [...map.querySelectorAll('.map-node')];
    const hits = cards.filter(c => {
      const r = c.getBoundingClientRect();
      return !(pr.right <= r.left || pr.left >= r.right || pr.bottom <= r.top || pr.top >= r.bottom);
    });
    const active = cards.find(c => c.classList.contains('trace-active'));
    const ar = active && active.getBoundingClientRect();
    const arrow = parseFloat(getComputedStyle(payload).getPropertyValue('--arrow-x')) || 0;

    samples.push({
      card: active ? (active.querySelector('.module-head b') || {}).textContent : '-',
      overlaps: hits.length,
      gapBelowCard: ar ? Math.round(pr.top - ar.bottom) : null,   // 正数=在卡片下方
      arrowOnCard: ar ? (pr.left + arrow >= ar.left - 3 && pr.left + arrow <= ar.right + 3) : false,
      mapOverflowX: map.scrollWidth - map.clientWidth,
    });
  }

  const gaps = samples.map(s => s.gapBelowCard).filter(g => g !== null);
  return {
    samples: samples.length,
    overlappingDocks: samples.filter(s => s.overlaps > 0).length,   // 期望 0
    minGap: gaps.length ? Math.min(...gaps) : null,                 // 期望 > 0
    arrowOnCard: `${samples.filter(s => s.arrowOnCard).length}/${samples.length}`,
    maxMapOverflow: Math.max(...samples.map(s => s.mapOverflowX)),
    detail: samples,
  };
}

/* ============ sticky 自检 ============ */

/**
 * sticky 极易被祖先的 overflow 悄悄破坏（见 references/layout-pitfalls.md）。
 * 直接滚动几个位置，看它是否真的钉住不动。
 */
async function auditSticky(selector = '.trace-console') {
  const el = document.querySelector(selector);
  if (!el) return { error: `未找到 ${selector}` };
  const doc = document.documentElement;
  const before = doc.scrollTop;
  const tops = [];
  for (const y of [400, 800, 1200, 1600]) {
    doc.scrollTop = y;
    await sleep(60);
    tops.push(Math.round(el.getBoundingClientRect().top));
  }
  doc.scrollTop = before;
  const pinned = new Set(tops).size === 1;          // 钉住时每次采样应完全相同
  return { position: getComputedStyle(el).position, tops, pinned };
}

if (typeof window !== 'undefined') {
  Object.assign(window, { auditPage, auditPayload, auditSticky });
  console.log('已就绪: await auditPage() / auditPayload() / auditSticky()');
}
