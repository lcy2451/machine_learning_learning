# 公平性

本目录收录 Google 机器学习课程的收入预测与公平性练习，以及基于课程编写的个人练习。

**运行环境说明：`google_fairness_baseline.py` 和 `google_fairness_mindiff.py` 在 macOS（Mac）环境下，使用该环境中的依赖库完成。它们在 Windows 环境运行时遇到过依赖库问题，目前尚未完成 Windows 兼容性验证。此说明仅针对这两个 Google 练习脚本。**

## 文件说明

| 文件 | 内容 |
| --- | --- |
| [google_fairness_baseline.py](./google_fairness_baseline.py) | 训练基础收入分类模型，执行整体与性别分组评估，生成基础模型报告 |
| [google_fairness_mindiff.py](./google_fairness_mindiff.py) | 独立训练基础模型，再使用 MinDiff 继续训练，生成基础模型与 MinDiff 报告 |
| [mlp_torch_11f.py](./mlp_torch_11f.py) | 个人 PyTorch 收入分类练习，包含训练、验证、测试及 ONNX 导出和读取预测代码 |
| [eval_torch_11f.py](./eval_torch_11f.py) | 独立评估文件的起始模板，目前仅有说明注释 |
| [models/mlp_torch_11f.onnx](./models/mlp_torch_11f.onnx) | 已导出的收入分类模型，输出 logits |
| [fairness_report.html](./fairness_report.html) | 基础模型的分组指标与混淆矩阵，支持切换分类阈值 |
| [min_diff_fairness_report.html](./min_diff_fairness_report.html) | MinDiff 模型的分组指标，支持切换分类阈值 |

原始课程的 Colab 链接保留在两个 `google_fairness_*.py` 文件顶部的注释中。

## 个人 PyTorch 练习

`mlp_torch_11f.py` 参考 `google_fairness_baseline.py`，使用 PyTorch 实现 `11 → 64 → 32 → 1` 的二分类网络。文件名中的 `mlp` 表示多层感知机，`torch` 表示 PyTorch，`11f` 表示 11 个输入特征。

代码按标签分层划分 80% 训练集、10% 验证集和 10% 测试集；验证集、测试集复用训练集的均值和标准差。训练采用 BCE with Logits、Adam（学习率 `0.001`）、批大小 `100` 和 `10` 轮训练，并计算整体 Loss、Accuracy 和 AUC。尚未实现性别分组评估和可视化。

当前训练和 ONNX 导出调用已注释，直接运行会加载已有 ONNX，逐条预测测试集前 50 条样本，打印高收入概率、预测类别、真实标签及判断是否正确。当前模型导出时没有设置动态批大小，输入固定为 `(1, 11)`。模型不包含标准化，脚本通过相同的数据及划分重新计算训练集的均值、标准差；这些参数和特征顺序尚未单独保存。

训练代码优先使用 CUDA，CUDA 不可用时回退到 CPU；模型、数据和评估指标放在同一设备上。已在 Windows、NVIDIA GeForce RTX 4060 Laptop GPU 上完成训练。

ONNX 预测设备由 ONNX Runtime 的 `providers` 单独指定，与 PyTorch 训练设备分开配置。当前加载代码使用 `CPUExecutionProvider` 在 CPU 上预测；如改用 `CUDAExecutionProvider`，需要 ONNX Runtime 的对应 GPU 依赖，当前 Windows 环境曾遇到 CUDA 执行提供程序加载失败的问题。

开发时使用本机 Anaconda 的 `machine_learning` 环境，该环境不包含在仓库中。运行个人 PyTorch 练习需自行准备兼容的 Python 环境及 pandas、PyTorch、TorchMetrics、scikit-learn 和 ONNX Runtime 等依赖；重新导出 ONNX 还需相应的导出依赖。本地数据集统一放在根目录 `data` 文件夹。`eval_torch_11f.py` 目前仍为注释模板，读取预测逻辑位于训练文件内。

下文的数据处理、模型结构、报告及运行说明均针对现有的两个 Google 练习脚本。

## 数据与预测任务

两个脚本均读取根目录下的 [acsincome_raw_2018.csv](../data/acsincome_raw_2018.csv)。路径根据脚本所在位置计算，不依赖终端的当前目录。

任务是预测个人年收入是否超过 50,000 美元。代码将 `PINCP` 收入列转换为二元标签：

- `PINCP > 50000`：标签为 `1`，表示高收入。
- `PINCP <= 50000`：标签为 `0`。

其余 11 列均作为模型输入，包括 `SEX` 性别列。公平性评估也使用该列分组：`1.0` 为男性（Male），`2.0` 为女性（Female）。分组分析时才将数值转换为可读名称，训练输入保留原始数值。

## 基础模型

模型使用 TensorFlow / Keras 构建，结构为：

```text
11 个输入特征 → 归一化 → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
```

最后一个节点输出收入超过 50,000 美元的预测概率。训练使用 Adam 优化器和二元交叉熵损失，监控准确率与 AUC。

当前参数：

| 参数 | 值 |
| --- | --- |
| 训练集 / 测试集 | 80% / 20% |
| 训练集抽样随机种子 | `200` |
| 训练批大小 | `100` |
| 基础模型训练轮数 | `10` |

训练结束后，脚本在测试集上评估整体表现，生成预测概率，并通过 TensorFlow Model Analysis（TFMA）计算整体和按性别分组的指标。

## MinDiff 训练

`google_fairness_mindiff.py` 包含完整的基础模型流程，可以单独运行，不需要先运行基础模型脚本，也不读取已保存的模型。

基础模型训练完成后，脚本从训练集中分别选出真实标签为 `1` 的女性和男性样本，作为 MinDiff 的两组辅助数据。它们各自按 `50` 条组成批次，不足一批的部分被舍弃。

代码使用 `MinDiffModel` 包装已经训练过的基础模型，采用 `MMDLoss`、`loss_weight=1`，再训练 `10` 轮。训练同时考虑收入分类损失和两组正例的预测分布差异，目标是缓解与召回率差异有关的偏差。

这里的辅助样本只包含真实高收入者，因此该训练目标不保证准确率、精确率、召回率和假阳性率都同时接近，也不保证每次训练都有相同的改善结果。

## 报告怎么看

两个 HTML 报告均提供 `0.10` 到 `0.90`、步长为 `0.05` 的阈值滑块，默认阈值为 `0.50`。预测概率大于或等于阈值时，报告将该样本判为高收入。

| 指标 | 在本练习中的含义 |
| --- | --- |
| Accuracy（准确率） | 该组所有样本中，收入类别判断正确的比例 |
| Precision（精确率） | 该组被预测为高收入的人中，实际高收入者的比例 |
| Recall（召回率） | 该组实际高收入者中，被模型识别出来的比例 |
| FPR（假阳性率） | 该组实际非高收入者中，被误判为高收入的比例 |

基础模型报告还显示男女各自的混淆矩阵，包含正确判断的低收入者（TN）、误判为高收入者（FP）、漏判的高收入者（FN）以及正确识别的高收入者（TP）。MinDiff 报告仅显示指标柱状图。

滑块只是用已有预测概率重新计算指标，不会重新训练模型。报告中某项指标的分母为零时，当前代码将该指标显示为 `0`。

## 运行方式

以下命令用于运行两个 Google 练习脚本，已运行的环境为 macOS。仓库不包含开发者本机的 Conda 环境，目前也未提供可直接复现 Python 环境的配置文件。运行前需自行准备兼容的 Python 环境及下文列出的依赖。

在仓库根目录执行。下面的 `machine_learning` 仅为开发者本机的环境名，请替换为自己已配置好的 Conda 环境名；如果使用其他环境管理方式，先按对应方式激活环境，再执行 Python 命令：

```shell
conda activate machine_learning
python 公平性/google_fairness_baseline.py
```

运行包含 MinDiff 的完整练习：

```shell
python 公平性/google_fairness_mindiff.py
```

脚本使用 NumPy、pandas、Plotly、TensorFlow、TensorFlow Model Analysis、TensorFlow Model Remediation 和 protobuf。基础模型脚本也导入了 Model Remediation。MinDiff 脚本在导入 TensorFlow 前设置 `TF_USE_LEGACY_KERAS=1`，使用旧版 Keras 接口；运行环境需要具备对应的兼容依赖。具体兼容版本未在本目录锁定。

每次运行都会重新训练并覆盖对应的 HTML 报告，然后尝试在浏览器中打开。MinDiff 脚本会先覆盖 `fairness_report.html`，再生成 `min_diff_fairness_report.html`。只查看仓库已有报告时，直接用浏览器打开 HTML 文件即可，无需运行训练脚本。

## 当前实现说明

- 归一化层目前在划分训练集和测试集之前，使用全部样本计算统计量，因此测试集信息参与了预处理，评估结果存在这一限制。
- 代码只固定了训练集抽样种子，没有固定全部训练随机性；两个脚本分别运行时，基础模型结果可能不同。比较 MinDiff 前后表现时，可查看 MinDiff 脚本在同一次运行中生成的两份报告。
- 当前脚本不保存训练后的模型，HTML 报告展示的是生成报告时的评估结果。
