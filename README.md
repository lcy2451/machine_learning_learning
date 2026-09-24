# 机器学习学习与实践

个人机器学习学习仓库，用于整理学习笔记、算法实现和练习代码。

## 内容入口

- [线性回归练习](./线性回归/README.md)：PyTorch 训练、绘图及 ONNX 导出与预测。
- [车费预测桌面应用](./taxi-app/README.md)：React + Electron 界面，使用现有 ONNX 模型离线预测车费，支持英里／公里和系统主题。

## 开发环境

以下为 Python 机器学习练习环境；桌面 App 使用独立的 Node.js/npm 项目，见下方介绍。

- 操作系统：Windows
- Python 发行版与环境管理：Anaconda / Conda
- Conda 环境：`machine_learning`
- Python 版本：`3.13.15`

## 主要库

以下为当前学习环境中已安装的主要库，具体依赖以各练习实际使用为准。

| 库 | 版本 | 用途 |
| --- | --- | --- |
| NumPy | 2.5.2 | 数值计算 |
| pandas | 3.0.5 | 数据处理与分析 |
| Matplotlib | 3.11.0 | 数据可视化 |
| Plotly | 6.9.0 | 交互式图表 |
| PyTorch（torch） | 2.10.0 | 张量计算与深度学习 |
| TensorFlow | 2.21.0 | 机器学习与深度学习 |
| Keras | 3.15.0 | 神经网络建模 |
| ONNX | 1.22.0 | 模型交换格式 |
| ONNX Runtime | 1.24.4 | 模型推理 |
| google-ml-edu | 0.1.3 | 机器学习课程辅助工具 |

环境信息记录于 2026-09-20。

## 车费预测 App

`taxi-app` 是一个 Windows 桌面学习应用，将训练好的单变量线性回归 ONNX 模型用于实际交互：输入行程里程，点击按钮或按回车，显示预测车费（美元）。支持英里／公里切换、系统浅色／深色主题、输入检查和模型加载失败提示，全部预测在本机离线完成。

界面使用 React + TypeScript 和普通 CSS，桌面框架为 Electron，主进程通过 ONNX Runtime Node.js 调用模型。应用只执行预测，不读取训练数据或重新训练。当前模型输入 14.4 英里时预测约为 **37.84 美元**；这是学习模型的估算值，不是实际报价。

安装项目依赖后，在仓库根目录启动：

```powershell
cd taxi-app
npm run dev
```

使用 `npm run pack` 可生成 Windows x64 免安装 exe，目标电脑无需安装 Python 或 Node.js。模型在打包时复制进程序，重新训练不会自动更新已生成的 exe，需要重新打包。

仓库保存 App 源码、依赖锁文件和测试，不保存 `node_modules`、构建目录或 exe。完整的依赖安装、模型接口、测试和打包说明见 [App README](./taxi-app/README.md)。
