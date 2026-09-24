import { app, BrowserWindow, ipcMain, Menu } from 'electron';
import path from 'node:path';
import { Predictor } from './predictor';

const predictor = new Predictor();
let ready: Promise<{ ready: boolean; error?: string }>;

function createWindow() {
  const window = new BrowserWindow({
    width: 800, height: 600, minWidth: 420, minHeight: 520,
    title: '车费预测', backgroundColor: '#101824',
    webPreferences: { preload: path.join(__dirname, 'preload.js'), contextIsolation: true, nodeIntegration: false, sandbox: true },
  });
  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  window.webContents.on('will-navigate', event => event.preventDefault());
  void window.loadFile(path.join(__dirname, '../dist/index.html'));
}

app.whenReady().then(() => {
  Menu.setApplicationMenu(null);
  // 开发时读取仓库原模型；打包后读取 exe 解压资源中的模型。
  const modelPath = app.isPackaged
    ? path.join(process.resourcesPath, 'models/lr_torch_1f.onnx')
    : path.resolve(__dirname, '../../线性回归/models/lr_torch_1f.onnx');
  ready = predictor.load(modelPath).then(() => ({ ready: true })).catch(error => {
    console.error('模型加载失败', error);
    return { ready: false, error: '模型加载失败，请检查应用文件是否完整，然后重新打开应用。' };
  });
  ipcMain.handle('model:status', () => ready);
  ipcMain.handle('model:predict', async (_event, input: unknown) => {
    const status = await ready;
    if (!status.ready) return { ok: false, error: status.error };
    try { return { ok: true, ...(await predictor.predict(input)) }; }
    catch (error) {
      console.error('预测失败', error);
      return { ok: false, error: error instanceof Error && /里程|输入|请选择|数值范围|准备好/.test(error.message)
        ? error.message : '预测失败，请重新输入后再试。' };
    }
  });
  createWindow();
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});
app.on('window-all-closed', () => app.quit());
