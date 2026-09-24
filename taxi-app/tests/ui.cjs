const { _electron: electron, expect } = require('@playwright/test');
const path = require('node:path');
const fs = require('node:fs');

(async () => {
  const env = { ...process.env };
  delete env.ELECTRON_RUN_AS_NODE;
  const app = await electron.launch({ args: [path.resolve(__dirname, '..')], env });
  try {
    const page = await app.firstWindow();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await expect(page.getByText('本地模型已就绪')).toBeVisible();
    const input = page.getByLabel('行程里程', { exact: true });
    const button = page.getByRole('button', { name: '预测车费' });
    await expect(button).toBeDisabled();
    await input.fill('3');
    await input.press('Enter');
    await expect(page.locator('.amount')).toHaveText('$11.86');
    await input.fill('1');
    await expect(page.locator('.amount')).toHaveText('— —');
    await page.getByRole('button', { name: '公里', exact: true }).click();
    await input.fill('1.609344');
    await button.click();
    await expect(page.locator('.amount')).toHaveText('$7.30');
    await input.fill('-1');
    await expect(button).toBeDisabled();
    await input.fill('0');
    await button.click();
    await expect(page.locator('.amount')).toHaveText('$5.02');
    await input.fill('1e40');
    await button.click();
    await expect(page.getByRole('alert')).toContainText('里程太大');
    await input.fill('3');
    await page.getByRole('button', { name: '英里', exact: true }).click();
    await button.click();
    await expect(page.locator('.amount')).toHaveText('$11.86');
    fs.mkdirSync(path.resolve(__dirname, '../test-artifacts'), { recursive: true });
    for (const colorScheme of ['light', 'dark']) {
      await page.emulateMedia({ colorScheme });
      await page.screenshot({ path: path.resolve(__dirname, `../test-artifacts/${colorScheme}.png`) });
    }
    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(440, 650));
    await expect(button).toBeVisible();
    await page.screenshot({ path: path.resolve(__dirname, '../test-artifacts/narrow.png') });
    // 用延迟响应确认连续提交只触发一次，且旧结果不会覆盖新输入。
    await app.evaluate(({ ipcMain }) => {
      globalThis.testCalls = 0;
      ipcMain.removeHandler('model:predict');
      ipcMain.handle('model:predict', async () => {
        globalThis.testCalls++;
        await new Promise(resolve => setTimeout(resolve, 400));
        return { ok: true, fare: 99, miles: 3 };
      });
    });
    await page.locator('form').evaluate(form => {
      form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
      form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    });
    await input.fill('4');
    await expect(button).toBeEnabled();
    await expect(page.locator('.amount')).toHaveText('— —');
    expect(await app.evaluate(() => globalThis.testCalls)).toBe(1);
    await app.evaluate(({ ipcMain }) => {
      ipcMain.removeHandler('model:predict');
      ipcMain.handle('model:predict', () => ({ ok: false, error: '预测失败，请重新输入后再试。' }));
    });
    await button.click();
    await expect(page.getByRole('alert')).toContainText('预测失败');
    await app.evaluate(({ ipcMain }) => {
      ipcMain.removeHandler('model:status');
      ipcMain.handle('model:status', () => ({ ready: false, error: '模型加载失败，请检查应用文件是否完整。' }));
    });
    await page.reload();
    await expect(page.getByRole('alert')).toContainText('模型加载失败');
    await expect(button).toBeDisabled();
    if (errors.length) throw new Error(errors.join('\n'));
    console.log('Electron UI: prediction, units, validation, keyboard, themes, resize, duplicate/stale requests and failure states passed');
  } finally { await app.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
