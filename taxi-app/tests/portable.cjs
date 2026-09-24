// 在独立临时目录验证 portable exe；远程调试端口仅用于此次自动化测试。
const { chromium, expect } = require('@playwright/test');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const net = require('node:net');

(async () => {
  const server = net.createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  await new Promise(resolve => server.close(resolve));
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'taxi-portable-'));
  const exe = path.join(temp, 'TaxiFare.exe');
  fs.copyFileSync(path.resolve(__dirname, '../release/TaxiFare-1.0.0-win-x64.exe'), exe);
  const env = { ...process.env, PATH: process.env.SystemRoot + '\\System32' };
  delete env.ELECTRON_RUN_AS_NODE;
  const child = spawn(exe, [`--remote-debugging-port=${port}`], { cwd: temp, env, stdio: 'ignore' });
  child.on('error', console.error);
  let browser;
  try {
    for (let attempt = 0; attempt < 120; attempt++) {
      try {
        const response = await fetch(`http://127.0.0.1:${port}/json/version`);
        const version = await response.json();
        browser = await chromium.connectOverCDP(version.webSocketDebuggerUrl);
        break;
      } catch { await new Promise(resolve => setTimeout(resolve, 1000)); }
    }
    if (!browser) throw new Error('免安装应用启动超时');
    const context = browser.contexts()[0];
    // 模拟应用页面离线，不改变用户机器的网络配置。
    await context.setOffline(true);
    const page = context.pages()[0];
    await page.reload();
    await expect(page.getByText('本地模型已就绪')).toBeVisible();
    await page.getByLabel('行程里程', { exact: true }).fill('3');
    await page.getByRole('button', { name: '预测车费' }).click();
    await expect(page.locator('.amount')).toHaveText('$11.86');
    console.log('Portable smoke passed: copied exe, isolated cwd/PATH, offline renderer, prediction $11.86');
    fs.mkdirSync(path.resolve(__dirname, '../test-artifacts'), { recursive: true });
    await page.screenshot({ path: path.resolve(__dirname, '../test-artifacts/portable.png') });
    await page.close();
  } finally {
    if (browser) await browser.close();
    child.kill();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
