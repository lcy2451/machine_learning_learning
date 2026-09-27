# 分类

本目录收录基于 Keras 的大米品种二分类练习，包括三特征模型、七特征模型，以及两个模型在验证集上的比较和七特征模型的测试集评估。

## 代码说明

| 文件 | 实现库 | 输入特征数量 | 输入特征 | 主要内容 |
| --- | --- | --- | --- | --- |
| [classification_example_1.py](./classification_example_1.py) | Keras、ml_edu | 3 | 偏心率、长轴长度、面积 | 训练基线模型，绘制指标曲线，比较训练集与验证集指标 |
| [classification_example_2.py](./classification_example_2.py) | Keras、ml_edu | 7 | 数据集中的全部七个数值特征 | 训练全特征模型，绘制指标曲线，比较训练集与验证集指标 |
| [classification_example_3.py](./classification_example_3.py) | Keras、ml_edu | 7 和 3 | 分别使用全部七个特征和基线的三个特征 | 比较两个模型的验证集指标，并评估七特征模型的测试集表现 |

三个脚本均预测米粒属于 `Cammeo` 还是 `Osmancik`。文件名末尾的 `1`、`2`、`3` 是练习编号，不代表输入特征数量。

## 数据与预处理

使用 [大米品种数据](../data/Rice_Cammeo_Osmancik.csv)，每行表示一粒大米的图像测量结果。模型输入是已经提取好的数值特征，不直接读取图片。

| 特征 | 含义 | 三特征模型 | 七特征模型 |
| --- | --- | --- | --- |
| `Eccentricity` | 偏心率，描述形状的细长程度 | 使用 | 使用 |
| `Major_Axis_Length` | 长轴长度 | 使用 | 使用 |
| `Minor_Axis_Length` | 短轴长度 | — | 使用 |
| `Area` | 面积 | 使用 | 使用 |
| `Convex_Area` | 凸包面积 | — | 使用 |
| `Perimeter` | 周长 | — | 使用 |
| `Extent` | 米粒面积与包围矩形面积的比值 | — | 使用 |

长度和周长以像素（px）为单位，面积以平方像素（px²）为单位。三个脚本都会输出长轴长度的最小值和最大值、面积范围，以及最大周长距离平均周长多少个标准差。

### 标准化与标签

数值特征使用 Z-score 标准化：

```text
标准化后的特征 = (原始特征 - 该特征的平均值) / 该特征的标准差
```

标准化后的数值表示原始值距离平均值多少个标准差。类别列 `Class` 不参与标准化，而是转换为标签列 `Class_Bool`：

- `Cammeo` → `1`，作为正类。
- `Osmancik` → `0`，作为负类。

准备输入时会去掉 `Class` 和 `Class_Bool`，防止把答案当作特征交给模型。训练函数再根据 `settings.input_features` 选取实际使用的特征。

### 数据划分

数据通过 `sample(frac=1, random_state=100)` 打乱，再按约 80%、10%、10% 分为训练集、验证集和测试集。Keras 随机种子设置为 `42`。

| 数据子集 | 用途 | 当前使用情况 |
| --- | --- | --- |
| 训练集 | 学习模型的权重和偏置 | 三个脚本均使用 |
| 验证集 | 评估训练后的模型、比较不同模型 | 三个脚本均使用 |
| 测试集 | 在模型比较之后进行最终评估 | 仅第三个脚本调用测试集评估 |

当前代码先用全部数据计算均值和标准差，再划分数据，因此验证集和测试集也参与了标准化统计量的计算。这里记录的是现有实现，评估结果需要结合这一点理解。

## 模型结构与训练方式

三个脚本使用相同的模型结构：每个特征对应一个 `keras.Input(shape=(1,))`，通过 `Concatenate` 拼接，再接一个带 Sigmoid 激活函数的 `Dense(units=1)` 输出层，没有隐藏层。这相当于逻辑回归二分类模型：

```text
z = 权重1 × 特征1 + 权重2 × 特征2 + … + 偏置
预测为 Cammeo 的概率 = sigmoid(z) = 1 / (1 + exp(-z))
```

模型输出介于 0 和 1 之间，再通过分类阈值判断类别。阈值影响 accuracy、precision、recall 的计算，不会改变训练使用的二元交叉熵损失。

| 配置 | 当前值 |
| --- | --- |
| 训练轮数 | 60 |
| 批次大小 | 100 |
| 学习率 | 0.001 |
| 优化器 | RMSprop |
| 损失函数 | `keras.losses.BinaryCrossentropy()`（二元交叉熵） |
| AUC 的阈值数量 | 100 |

`create_model()` 负责创建和编译模型；`train_model()` 把选定的特征转换成 NumPy 数组字典，交给 `model.fit()` 训练，并将模型、配置和训练历史保存到内存中的 `ml_edu.experiment.Experiment` 对象。

当前 `fit()` 没有传入验证集，曲线展示的是每轮训练指标；验证集通过训练结束后的 `evaluate()` 单独评估。三个脚本均未保存或导出模型文件。

## classification_example_1.py：三特征基线模型

使用偏心率 `Eccentricity`、长轴长度 `Major_Axis_Length` 和面积 `Area` 训练名为 `baseline` 的实验。模型包含三个权重和一个偏置，分类指标的阈值为 `0.35`。

结果展示包括：

- accuracy、precision、recall 随训练轮数变化的曲线。
- AUC 随训练轮数变化的曲线。
- 训练集最后一轮指标与训练后验证集指标的逐项打印比较。

脚本准备了测试集特征和标签，但没有调用测试集评估。

## classification_example_2.py：七特征模型

使用全部七个数值特征训练名为 `all features` 的实验。模型包含七个权重和一个偏置，分类指标的阈值为 `0.5`。

脚本绘制七特征模型的 accuracy、precision、recall 和 AUC 曲线，并打印训练集与验证集指标。它只训练一个七特征模型，没有在同一次运行中训练三特征模型或比较两个模型，也没有评估测试集。

当前文件末尾额外两次绘图引用了未定义的 `experiment`，而本文件创建的实验变量名是 `experiment_all_features`。独立运行到这里会触发 `NameError`；前面的全特征模型训练、绘图和验证集评估代码位于这两行之前。

## classification_example_3.py：模型比较与测试集评估

在同一份数据划分上依次训练七特征模型和三特征基线模型，主要流程为：

```text
数据预处理与划分 → 训练七特征模型 → 绘图与验证集评估
→ 训练三特征模型 → 比较两个模型的验证集 accuracy、AUC
→ 输出七特征模型的测试集指标
```

`ml_edu.results.compare_experiment()` 接收两个实验、要比较的 `accuracy` 和 `auc`，以及验证集特征和标签。测试集评估则明确调用 `experiment_all_features.evaluate()`，因此当前评估的是七特征模型，代码没有根据比较结果自动选择模型。

当前配置有两个需要区分的地方：

- 七特征实验设置的分类阈值为 `0.5`，三特征实验设置为 `0.35`；但创建三特征模型时复用了按 `0.5` 构造的同一组 Keras 指标对象，因此其编译指标的阈值仍为 `0.5`，不能仅根据实验配置将其解释为按 `0.35` 计算。
- 三特征模型训练后的 `compare_train_validation()` 调用仍传入七特征实验，因此这次打印比较的仍是七特征模型的训练集和验证集指标。两个模型的验证集比较由后面的 `compare_experiment()` 完成。

脚本中的训练曲线绘图调用针对七特征模型，末尾逐项打印该模型的测试集指标。

## 分类指标说明

以 `Cammeo` 为正类，代码使用以下指标：

| 指标 | 含义 |
| --- | --- |
| `accuracy`（准确率） | 全部米粒中，类别判断正确的比例 |
| `precision`（精确率） | 被模型判为 Cammeo 的米粒中，实际为 Cammeo 的比例 |
| `recall`（召回率） | 实际为 Cammeo 的米粒中，被模型正确找出的比例 |
| `auc`（ROC AUC） | 综合多个分类阈值衡量模型区分两个类别的能力，不只看某一个固定阈值 |

损失函数负责指导参数更新，评估指标用于观察分类表现。`compare_train_validation()` 比较的是训练历史中最后一轮的指标与训练结束后在验证集上计算的指标。

## 运行方式

在项目根目录打开终端，使用已有的 Anaconda `machine_learning` 环境：

```powershell
conda activate machine_learning
cd 分类
python classification_example_1.py
```

将文件名替换为 `classification_example_2.py` 或 `classification_example_3.py` 即可运行对应练习；第二个脚本末尾的当前错误见上文说明。

数据通过 `../data/Rice_Cammeo_Osmancik.csv` 读取，因此运行时工作目录需要是 `分类`。脚本导入 Keras、ml_edu、NumPy、pandas 和 Plotly，并需要当前环境中可用的 Keras 后端。

三个文件的训练流程都直接写在模块顶层，直接运行会重新训练，导入这些模块也会执行数据读取和训练。
