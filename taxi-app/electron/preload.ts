import { contextBridge, ipcRenderer } from 'electron';

// 只开放两个固定接口，不把文件系统或通用 IPC 能力交给网页。
contextBridge.exposeInMainWorld('taxi', {
  status: () => ipcRenderer.invoke('model:status'),
  predict: (input: { distance: number; unit: 'mi' | 'km' }) => ipcRenderer.invoke('model:predict', input),
});
