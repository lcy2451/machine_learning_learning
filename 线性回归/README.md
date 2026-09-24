# 线性回归

本目录收录基于 PyTorch 的线性回归练习，包括单变量、多变量、公司字段编码实验，以及 ONNX 模型导出与读取预测。

## 代码说明

| 文件 | 实现库 | 输入特征数量 | 输入特征 | 预测目标 |
| --- | --- | --- | --- | --- |
| [lr_torch_1f.py](./lr_torch_1f.py) | PyTorch（torch） | 1（单变量） | 行程里程 `TRIP_MILES` | 车费 `FARE` |
| [lr_torch_2f.py](./lr_torch_2f.py) | PyTorch（torch） | 2（多变量） | 里程 `TRIP_MILES`、时长 `TRIP_SECONDS` | 车费 `FARE` |
| [lr_torch_3f.py](./lr_torch_3f.py) | PyTorch（torch） | 当前为 2 | 里程 `TRIP_MILES`、时长 `TRIP_SECONDS`；公司列已预处理但未加入输入 | 车费 `FARE` |
| [lr_torch_1f_onnx.py](./lr_torch_1f_onnx.py) | PyTorch 训练与导出、ONNX Runtime 预测 | 1（单变量） | 行程里程 `TRIP_MILES` | 车费 `FARE` |

文件名 `lr_torch_1f.py` 中，`lr` 表示线性回归，`torch` 表示 PyTorch，`1f` 表示 1 个输入特征。

`lr_torch_3f.py` 保留公司字段实验的文件名，当前实际输入数量以代码中的 `_in_feature_names` 为准，不能仅根据文件名判断。

导出的模型文件为 [models/lr_torch_1f.onnx](./models/lr_torch_1f.onnx)，由 `lr_torch_1f_onnx.py` 保存和读取，不是单独的 Python 脚本。

## lr_torch_1f.py：PyTorch 单变量线性回归

使用 [芝加哥出租车数据](../data/chicago_taxi_train.csv)，学习行程里程与车费之间的线性关系。pandas 负责读取数据，PyTorch 负责训练，Matplotlib 负责绘制结果。

模型由 `torch.nn.Linear(1, 1)` 定义：

```text
预测车费 = 权重 × 行程里程 + 偏置
```

### 训练方式

`train()` 接收输入张量、目标张量、损失函数、绘图数据以及训练轮数、批次大小和学习率。通过 `TensorDataset` 和 `DataLoader` 将数据分批并打乱顺序，使用反向传播与 RMSprop 优化器更新模型参数。

| 配置 | 当前值 |
| --- | --- |
| 训练轮数 | 20 |
| 批次大小 | 50 |
| 学习率 | 0.001 |
| 随机种子 | 42 |
| 优化器 | RMSprop |
| 当前损失函数 | `torch.nn.L1Loss()`（MAE） |

目前支持 `L1Loss()`（平均绝对误差）和 `MSELoss()`（均方误差），均使用默认的均值归约方式。损失函数从外部传入，训练时使用所选损失计算梯度，每轮结束后在整个训练集上计算损失和 RMSE。

RMSE 的定义为：

```text
RMSE = sqrt(mean((预测值 - 真实值)²))
```

使用 MSE 时，对训练集 MSE 开平方即可得到 RMSE；使用 MAE 时，另行根据预测误差计算 RMSE。

### 结果展示

脚本输出每轮损失、RMSE 以及模型权重和偏置，并绘制三张图：

- **训练损失**：展示所选损失随训练轮数的变化，纵轴随损失函数显示 MAE 或 MSE。
- **训练集 RMSE**：展示预测误差随训练轮数的变化，与车费使用相同单位。
- **拟合结果**：抽取最多 200 条样本绘制实际车费散点，在全部训练数据的里程范围内绘制预测直线，并用红色五角星标出 1 英里的预测车费。

当前代码使用全部数据训练和计算指标，没有单独划分测试集，因此图中反映的是训练集上的拟合表现。

数据文件通过 `../data/chicago_taxi_train.csv` 读取，运行时的工作目录需要设为本目录（`线性回归`）。

## lr_torch_2f.py：PyTorch 双变量线性回归

使用行程里程 `TRIP_MILES` 和行程时长 `TRIP_SECONDS` 共同预测车费 `FARE`。输入每行按 `[里程, 时长]` 排列，形状为 `[样本数, 2]`；模型为 `torch.nn.Linear(2, 1)`，包含两个权重和一个偏置：

```text
预测车费 = 里程权重 × 里程 + 时长权重 × 时长 + 偏置
```

模型在主程序中创建并传给 `train()`。当前使用 RMSprop、MAE 损失，训练 60 轮，批次大小为 50，学习率为 0.001。每轮计算整个训练集的损失和 RMSE，训练结束后打印两个权重和偏置。

结果展示包括三张图：

- 训练损失曲线。
- 训练集 RMSE 曲线。
- 三维真实行程散点和模型预测平面，三个坐标分别为里程、时长和车费。

三维散点从训练张量中随机抽取最多 200 条真实行程。预测平面在里程和时长的范围内各取 20 个值，组合成 400 组输入后进行预测；网格组合用于展示模型平面，不代表每组组合都对应真实行程。

该脚本使用全部数据训练和计算指标，不导出模型。数据路径同样为 `../data/chicago_taxi_train.csv`，运行时工作目录需设为 `线性回归`。

## lr_torch_1f_onnx.py：训练、保存与读取 ONNX 模型

该脚本基于单变量线性回归练习，使用里程预测车费，完整流程为：

```text
读取 CSV → PyTorch 训练 → 保存 ONNX 文件 → ONNX Runtime 加载 → 预测 14.4 英里的车费
```

`train()` 使用 `torch.nn.Linear(1, 1)` 和 RMSprop 训练，当前主程序配置为 60 轮、批次大小 50、学习率 0.001、MSE 损失。每轮打印训练集损失和 RMSE，结束后打印权重和偏置，并返回内存中的 PyTorch 模型。`save_onnx` 参数默认是 `False`，主程序显式传入 `True`，因此当前直接运行脚本会导出模型。当前脚本不绘图，保留的 `plot_df` 参数没有参与计算。

当前保存模型的权重约为 `2.279238`，偏置约为 `5.020588`，输入 14.4 英里时预测约为 `37.8416` 美元。这是根据全部训练数据拟合的直线结果，不是按里程查找某一条原始记录；相同里程的真实车费可以不同。

### 保存模型的代码

主程序先准备保存位置：

```python
_onnx_path = Path(__file__).resolve().parent / "models" / "lr_torch_1f.onnx"
_onnx_path.parent.mkdir(parents=True, exist_ok=True)
```

`__file__` 表示当前脚本，`resolve().parent` 得到脚本所在目录。模型保存到本目录的 `models/lr_torch_1f.onnx`，父目录不存在时自动创建。

`train()` 在训练结束后，仅当 `save_onnx=True` 时执行以下导出代码：

```python
model.eval()
example_input = torch.tensor([[1.0]], dtype=torch.float32)

torch.onnx.export(
    model,
    (example_input,),
    onnx_path,
    input_names=["TRIP_MILES"],
    output_names=["FARE"],
    dynamo=True,
    external_data=False,
)
```

- `model.eval()` 切换到评估模式，本身不执行预测或训练。
- 示例输入用于确定输入格式；`[[1.0]]` 的形状为 `[1, 1]`，表示一条行程、一个特征，不是限制只能预测 1 英里。
- `(example_input,)` 是单元素元组，表示模型接收一个输入张量。
- `TRIP_MILES` 和 `FARE` 是模型输入、输出的名字，读取时需要对应。
- `dynamo=True` 使用基于 `torch.export` 的 ONNX 导出方式。
- `export_params` 默认是 `True`，因此会保存已训练好的权重和偏置。`external_data=False` 表示参数放进 `.onnx` 文件内部，不拆成 `.onnx.data` 文件；这两个参数作用不同。

导出的文件包含预测计算及参数，即“车费 = 权重 × 里程 + 偏置”，不包含 CSV 数据、Python 训练循环或优化器状态。重新导出到同一路径会覆盖原模型。

### 读取模型并预测的代码

脚本的 `predict(onnx_path)` 函数执行以下操作：

```python
session = ort.InferenceSession(
    str(onnx_path),
    providers=["CPUExecutionProvider"],
)

x = np.array([[14.4]], dtype=np.float32)
outputs = session.run(["FARE"], {"TRIP_MILES": x})
fare = outputs[0][0, 0]
print(f"fare {fare:.4f}")
```

`InferenceSession` 从文件加载模型，创建使用 CPU 的推理会话。`session.run()` 将里程交给模型，使用已保存的参数计算车费，不会重新训练。

输入使用 NumPy 数组，类型 `np.float32` 与导出时的 `torch.float32` 对应。当前没有配置动态形状，输入固定为 `[1, 1]`，一次预测一条行程。把 `14.4` 换成其他里程即可预测其他行程。

`outputs` 是输出列表，`outputs[0]` 取出名为 `FARE` 的数组；该数组形状为 `[1, 1]`，再用 `[0, 0]` 取出第一条行程的预测车费。

### 运行方式与当前行为

在项目根目录打开终端，使用已有的 Anaconda `machine_learning` 环境：

```powershell
conda activate machine_learning
cd 线性回归
python lr_torch_1f_onnx.py
```

训练数据仍通过 `../data/chicago_taxi_train.csv` 读取，所以运行时工作目录需要是 `线性回归`。导出需要 `onnx` 和 `onnxscript`，读取预测需要 `onnxruntime`。

直接运行脚本会每次重新训练、保存，然后预测；`predict()` 函数本身只读取模型。已有模型时，也可以在本目录的 Python 会话中单独调用：

```python
from pathlib import Path
from lr_torch_1f_onnx import predict

predict(Path("models/lr_torch_1f.onnx"))
```

导入模块不会执行 `if __name__ == '__main__':` 中的数据读取和训练，但仍需要模块顶部导入的库。当前训练指标来自训练集，脚本尚未自动对比 PyTorch 与 ONNX 的预测结果。

## lr_torch_3f.py：公司字段编码练习

`lr_torch_3f.py` 用于探索行程里程 `TRIP_MILES`、行程时长 `TRIP_SECONDS` 和公司 `COMPANY` 与车费 `FARE` 的关系。当前为对比测试，`_in_feature_names` 仅选择里程和时长，实际训练使用两个输入。脚本仍保留公司预处理：名称先统一大写、去掉逗号并清理首尾空格，再将名称中每个字符的 Unicode 编码（`ord()`）相加，最终转换为浮点数。这种方式将公司信息保留为一列，不使用独热编码。

当前练习有意保留这一实现，接受以下局限：

- 不同名称可能得到相同的编码和，例如 `AB` 与 `BA` 都得到 `131`，因此名称去重不代表数值编码唯一。
- 编码的大小和差值不代表公司的收费关系。线性模型会用同一个公司权重乘以编码和，强制公司对预测值的影响随编码和线性变化，无法为每家公司独立学习调整金额。
- 因此，该方式不适合作为线性回归中公司类别的一般编码方案；目前仅作为自定义字符串转数值的学习实验，不据此认定预测效果有所改善。

后续围绕本练习检查代码时，将此编码方式视为已接受的限制，除非主动讨论编码方案，否则不重复建议替换。

当前脚本已启用训练，模型输入数量由 `_in_feature_names` 决定。绘图保留训练 Loss、RMSE 曲线，并为每个输入特征绘制一张车费对比图：从训练数据抽取最多 200 条真实行程，对这批行程进行预测，以该特征为横轴，用蓝色散点表示真实车费、橙色叉号表示预测车费。两组点逐行对应，不使用人工网格或预测连线。若将公司加入输入列表，绘图也会使用该列；显示的仍是训练数据上的拟合表现。

## 输入特征与模型参数

“一个输入”指一个输入特征，即单变量线性回归。包含偏置的单变量线性回归模型有两个需要训练的参数：一个权重和一个偏置。

多变量线性回归可以同时使用多个输入特征，例如行程里程和行程时长，其模型形式为：

```text
预测值 = 权重1 × 特征1 + 权重2 × 特征2 + … + 偏置
```

不同实现通过所用库和输入特征数量加以区分，新增练习可在上方表格中补充说明。
