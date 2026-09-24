# 车费预测桌面应用

React + TypeScript 编写界面，Electron 提供 Windows 窗口，ONNX Runtime Node.js 在主进程中使用 CPU 预测。应用直接读取现有单变量模型，不执行训练，不需要服务器。

## 使用

输入非负里程，选择英里或公里，点击“预测车费”或按回车。结果以美元显示两位小数。公里除以 `1.609344` 换算为英里；切换单位会按新单位解释输入，并清空旧结果。界面自动跟随系统浅色、深色主题。

这是基于芝加哥出租车数据的学习模型，结果不是实际出租车报价。应用不保存预测历史，不发送网络请求。

## 本地开发

需要 Node.js 22.12+ 或当前项目使用的 Node.js 24。经授权安装项目依赖后，在本目录运行：

```powershell
# 仅需 CPU 推理，跳过可选 CUDA 下载
$env:ONNXRUNTIME_NODE_INSTALL_CUDA = 'skip'
npm ci
npm run dev
```

`dev` 先构建再打开 Electron 窗口，不运行开发服务器；修改代码后重新执行即可。

```powershell
npm run check    # TypeScript 类型检查
npm test         # 构建并验证模型、单位换算和异常输入
npm run test:ui  # 启动真实 Electron 窗口进行交互检查
npm run pack     # Windows x64 免安装 exe
npm run test:portable # 打包后，在独立临时目录进行离线页面预测检查
```

构建产物位于 `release/TaxiFare-1.0.0-win-x64.exe`，可以复制到其他目录运行。免安装程序运行时会把内置资源解压到临时目录；无需用户安装 Python 或 Node.js。当前个人学习版不进行代码签名，不提供自动更新。

打包脚本复用 npm 安装的 Electron；首次打包仍可能需要下载 NSIS 和解压工具。若检测到 `Program Files/7-Zip/7z.exe`，会自动使用它；也可通过 `ELECTRON_BUILDER_7ZIP_PATH` 指定其他位置。

## 模型与代码

- `electron/predictor.ts`：输入验证、公里换算、模型加载及预测。会话在启动时加载一次，后续复用。
- `electron/main.ts`：窗口、资源路径和固定 IPC 接口；`preload.ts` 只开放状态查询与预测。
- `src/main.tsx`、`src/style.css`：界面、输入状态、结果和系统主题。
- `tests/predictor.test.cjs`：与 Python ONNX Runtime 基准值对比及错误场景。

开发时模型来自 `../线性回归/models/lr_torch_1f.onnx`。打包通过 `extraResources` 复制原文件，发布后从 `process.resourcesPath/models/lr_torch_1f.onnx` 加载。原生推理模块从 ASAR 解包，以便 Windows 加载相关 DLL。

当前模型使用 MSE、60 轮训练，3 英里预测约为 `$11.86`，14.4 英里约为 `$37.84`。重新训练后，开发模式重启即可读取新模型；已经生成的 exe 保留打包时的模型，必须重新执行 `npm run pack` 才能更新。测试基准也需与对应模型一致。

预测接口为 `predict({ distance: number, unit: 'mi' | 'km' })`。成功返回 `{ ok: true, fare, miles }`；失败返回 `{ ok: false, error }`。主进程始终重复验证输入，渲染进程不具有 Node.js 或文件系统权限。

模型输入为 `TRIP_MILES`，类型 `float32`，形状 `[1, 1]`；输出为 `FARE`。不进行人民币换算，也不对负预测值做自动截断。模型缺失、加载失败或数值溢出都会显示中文错误。

单元检查使用 0、1、3、10、14.4 英里的 Python 基准，误差阈值为 `atol=1e-5`、`rtol=1e-5`。这些数值对应仓库当前模型，更换模型后需要重新生成基准。跨机器兼容性需在对应 Windows 环境中另行验证。

交互检查覆盖回车、单位切换、零和负数、溢出、连续提交、旧请求结果、模型失败提示，以及两种主题和窄窗口截图。`test:portable` 将 exe 复制到系统临时目录，将 PATH 限制为 Windows 系统目录，并通过浏览器离线模拟检查预测；这不等于在干净 Windows 机器上测试，也不会断开用户电脑的网络。

2026-09-24 首次免安装包使用旧模型完成独立目录、受限 PATH 和页面离线模拟检查，当时 3 英里输出为 `$11.22`。该旧 exe 不代表当前仓库模型；更新模型后的测试以 `$11.86` 为准，免安装包测试需要先重新打包。尚未在另一台干净 Windows 电脑上验证。

源码、依赖锁文件及测试纳入 Git；依赖目录 `node_modules/`、构建输出 `dist/`、`dist-electron/`、`release/` 和测试产物 `test-artifacts/` 均被忽略。
