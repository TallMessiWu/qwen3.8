# 在本机（WSL）跑 `audit-browser.js`

`scripts/audit-browser.js` 的 `auditPage` / `auditPayload` / `auditSticky` 需要**真浏览器**
（要算 `scrollWidth`、要点开每个模块、要跟着动画跑），`node` 里跑不了。本机是 WSL，
装不上 Linux Chrome，但 **Windows 侧本来就有 Chrome 和 node**，借过来即可。

## 为什么不能直接在本机装

- `npx puppeteer browsers install chrome` 所有 provider 都失败；
- 手动从 npmmirror 拉 zip 能下下来，但起不来：缺 `libnspr4` / `libnss3` / `libasound2`，
  而 `sudo` 要密码。

别再试这两条路，直接走下面的链路。整条链路 2026-09-16 实测跑通全量审计。

## 链路

**① WSL 起静态服务**（Windows 能直接访问 WSL 的 localhost）：

```bash
cd <架构页所在目录>
python3 -m http.server 8123
```

**② 拉起 Windows Chrome**：

```bash
"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" \
  --headless=new --remote-debugging-port=9333 \
  --user-data-dir='C:\Temp\ccaudit-chrome' about:blank &
```

**③ 关键一步：加一个 TCP 中继。** Chrome 只肯绑 `127.0.0.1`
（`--remote-debugging-address=0.0.0.0` 被无视），WSL 够不着 Windows 的 localhost。
用 Windows 的 `node.exe` 跑一个 `0.0.0.0:9444 → 127.0.0.1:9333` 的转发（八行代码就够）：

```js
// ccrelay.js —— 用 Windows 的 node.exe 跑：node.exe ccrelay.js
const net = require('net');
net.createServer(c => {
  const s = net.connect(9333, '127.0.0.1');
  c.pipe(s); s.pipe(c);
  c.on('error', () => s.destroy()); s.on('error', () => c.destroy());
}).listen(9444, '0.0.0.0');
```

**④ WSL 侧连上去跑审计**：

```bash
npm i puppeteer        # 只用它的连接能力，不要它自带的浏览器
```

```js
const puppeteer = require('puppeteer');
const gw = require('child_process')
  .execSync("ip route | awk '/default/{print $3}'").toString().trim();  // WSL 默认网关 = Windows 宿主
const browser = await puppeteer.connect({ browserURL: `http://${gw}:9444` });
const page = await browser.newPage();
await page.goto('http://localhost:8123/<页面>.html');   // 页面走 WSL 的 8123
// 然后照常注入 scripts/audit-browser.js，调 auditPage() / auditPayload() / auditSticky()
```

## 纪律

- **先跑基线。** 审计前对 `git show HEAD:<page>` 导出的那份跑一遍，否则分不清报出来的
  溢出是不是自己这次引入的。
- **收尾务必清干净**（按 cmdline 过滤，别误杀别的 Chrome）：
  Windows 侧 `Stop-Process` 掉 cmdline 含 `ccaudit` / `ccrelay` 的 chrome 与 node；
  WSL 侧 `kill` 掉 `http.server`；删 `C:\Temp\ccaudit-*`——目录句柄可能还被占着，
  删不掉就留个空目录，无害。
- 审计报出的问题对号入座见 `layout-pitfalls.md`；**空数组 / 零计数才是通过**。
