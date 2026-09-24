const fs = require('node:fs');
const path = require('node:path');
const builder = require('electron-builder');

// 复用 npm 已安装的 Electron，避免打包时重复下载同一运行时。
const electronDist = path.dirname(require('electron'));
const system7zip = path.join(process.env.ProgramFiles || 'C:/Program Files', '7-Zip', '7z.exe');
if (!process.env.ELECTRON_BUILDER_7ZIP_PATH && fs.existsSync(system7zip)) {
  process.env.ELECTRON_BUILDER_7ZIP_PATH = system7zip;
}
builder.build({
  targets: builder.Platform.WINDOWS.createTarget(process.argv.includes('--dir') ? 'dir' : 'portable', builder.Arch.x64),
  config: { electronDist },
}).catch(error => { console.error(error); process.exitCode = 1; });
