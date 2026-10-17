"""个人公平性练习：参考 Google 基础模型编写 PyTorch 版本（11 个输入特征）。"""

# 参考文件：同目录下的 google_fairness_baseline.py。
# 数据集：仓库根目录 data/acsincome_raw_2018.csv。
# 预测任务：个人年收入是否超过 50,000 美元。
# 在这里逐步编写自己的练习代码。
# cpu训练耗时380.55 秒
# gpu训练训练总耗时: 644.48 秒

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch

# 安装 pytorch 指标 python -m pip install torchmetrics
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

from torch.utils.data import TensorDataset, DataLoader

import onnxruntime as ort

# 使用pip安装sklearn 用来分训练集， 验证集， 测试集 pip install -U scikit-learn
from sklearn.model_selection import train_test_split

def calculate_normalize(x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 每个特征单独计算平均值和标准差
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)

    # 防止标准差为 0
    std = std.clamp(min=1e-7)

    # 标准化
    return mean, std, (x - mean) / std

import time
import functools


def monitor_training(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        model = kwargs["model"]
        device = next(model.parameters()).device

        print(f"训练设备: {device}")

        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(device)}")
            torch.cuda.synchronize(device)

        start = time.perf_counter()

        result = func(*args, **kwargs)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - start

        print(f"训练总耗时: {elapsed:.2f} 秒")

        return result

    return wrapper

@monitor_training
def train(train_loader:DataLoader,
          val_loader:DataLoader,
          model:torch.nn.Module,
          loss_fn:Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
          optimizer:torch.optim.Optimizer,
          accuracy: BinaryAccuracy,
          auc: BinaryAUROC,
          epochs:int):

    # 开始按 Epoch 训练
    for epoch in range(epochs):
        # 模型切换到训练
        model.train()
        for x_batch, y_batch in train_loader:
            #
            # x_batch = x_batch.to(device)
            # y_batch = y_batch.to(device)

            # 1. 前向传播
            _logits = model(x_batch)

            # # 测试前向传播
            # with torch.no_grad():
            #     output = _model(_normalize[:5])

            # 2. 计算 Loss
            loss = loss_fn(_logits, y_batch)

            # 3. 清空梯度
            optimizer.zero_grad()

            # 4. 反向传播
            loss.backward()

            # 5. 更新参数
            optimizer.step()

        # 把模型切换到评估模式（Evaluation Mode）。
        model.eval()

        # 每个 Epoch 开始评估时重置指标
        accuracy.reset()
        auc.reset()

        # 关闭梯度记录
        with torch.no_grad():

            epoch_loss = 0
            for val_x_batch, val_y_batch in val_loader:
                # val_x_batch = val_x_batch.to(device)
                # val_y_batch = val_y_batch.to(device)

                # 用验证集验证
                predictions = model(val_x_batch)

                batch_loss = loss_fn(predictions, val_y_batch)
                epoch_loss += batch_loss * len(val_x_batch)

                # 更新指标
                probabilities = predictions.sigmoid().flatten()
                targets = val_y_batch.int().flatten()
                accuracy.update(probabilities, targets)
                auc.update(probabilities, targets)

            epoch_loss = epoch_loss / len(val_loader.dataset)

            # 汇总整个验证集的指标
            epoch_accuracy = accuracy.compute()
            epoch_auc = auc.compute()

            print(f"epoch: {epoch+1}/{epochs}, val loss: {epoch_loss:.4f},"
                  f" accuracy: {epoch_accuracy:.4f}, auc: {epoch_auc:.4f}")


if __name__ == '__main__':
    # 学习率
    _learning_rate = 0.001
    _batch_size = 100
    _epochs = 10

    # 随机种子， 42 是“生命、宇宙以及一切的终极答案”。
    torch.manual_seed(42)

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _csv = Path(__file__).parent.parent / 'data' / 'acsincome_raw_2018.csv'
    _acs_income_dataset = pd.read_csv(_csv)
    _LABEL_KEY = "PINCP"
    # PINCP 个人年总收入
    _training_df: pd.Series | pd.DataFrame = _acs_income_dataset.loc[
        :, (_LABEL_KEY, "AGEP",  "COW",  "SCHL",  "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P", "ST")
    ]
    # print(_training_df["PINCP"])


    # 任务是预测一位美国在职成年人的年收入是否超过50, 000美元。
    # 数据集中的 PINCP （个人年总收入）列代表目标变量；
    # 不过，该值需要转换为二值形式。对于每个样本，
    # 如果 PINCP > 50000.0 ，则个体的目标标签为1 ，否则为 0 。
    _training_df.loc[_training_df[_LABEL_KEY] <= 50000.0, _LABEL_KEY] = 0
    _training_df.loc[_training_df[_LABEL_KEY] > 50000.0, _LABEL_KEY] = 1

    # print(_training_df.sample(10, random_state=0))

    _features = _training_df.drop(columns=["PINCP"])
    # _train_loader = DataLoader(_dataset, batch_size=_batch_size, shuffle=True)
    # 划分 80训练集
    _train_df, _temp_df = train_test_split(
        _training_df, test_size=0.2, random_state=42, stratify=_training_df[_LABEL_KEY])

    # 划分验证集和测试集
    _val_df, _test_df = train_test_split(
        _temp_df, test_size=0.5, random_state=42, stratify=_temp_df[_LABEL_KEY]
    )

    _x: torch.Tensor = torch.tensor(_train_df.drop(columns=[_LABEL_KEY]).to_numpy(),
                                    dtype=torch.float32).to(_device)
    # 真实标签 PINCP 真实答案
    _y = torch.tensor(_train_df[_LABEL_KEY].to_numpy(),dtype=torch.float32).reshape(-1, 1).to(_device)

    _val_x: torch.Tensor = torch.tensor(_val_df.drop(columns=[_LABEL_KEY]).to_numpy(),dtype=torch.float32).to(_device)
    _y_val = torch.tensor(
        _val_df[_LABEL_KEY].to_numpy(),
        dtype=torch.float32
    ).reshape(-1, 1).to(_device)

    _test_x: torch.Tensor =torch.tensor(
        _test_df.drop(columns=[_LABEL_KEY]).to_numpy(),dtype=torch.float32).to(_device)
    _test_y = torch.tensor(
        _test_df[_LABEL_KEY].to_numpy(),
        dtype=torch.float32
    ).reshape(-1, 1).to(_device)


    _mean, _std, _normalize = calculate_normalize(_x)
    val_normalize = (_val_x - _mean) / _std
    test_normalize = (_test_x - _mean) / _std

    # nn.Linear：一个线性层。
    # nn.Sequential：一个容器，按照顺序把多个层连接起来。
    # binary_cross_entropy_with_logits已经在内部处理了Sigmoid，因此模型末尾不应该带Sigmoid。
    _model = torch.nn.Sequential(
        torch.nn.Linear(_x.shape[1], 64),
        torch.nn.ReLU(),

        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),

        torch.nn.Linear(32, 1),
        # torch.nn.Sigmoid(),
    )

    _model = _model.to(_device)

    # _logits = _model(_normalize)
    # 测试前向传播
    # with torch.no_grad():
    #     output = _model(_normalize[:5])
    #
    # print(output, output.shape)

    # 任务：预测年收入是否超过 50, 000 美元
    # 创建 损失函数 loos 使用 Binary Cross Entropy（二元交叉熵）。
    # _loss_fn = torch.nn.functional.binary_cross_entropy_with_logits(_logits, _y)
    _loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
    # _loss_fn = torch.nn.BCELoss()

    # 优化器创建
    _optimizer = torch.optim.Adam(_model.parameters(), lr=_learning_rate)

    # 定义指标
    _accuracy = BinaryAccuracy().to(_device)
    _auc = BinaryAUROC().to(_device)

    # 训练数据集
    _train_dataset = TensorDataset(_normalize, _y)
    _train_loader = DataLoader(_train_dataset, batch_size=_batch_size, shuffle=True)

    # 验证数据集
    _val_dataset = TensorDataset(val_normalize, _y_val)
    _val_loader = DataLoader(_val_dataset, batch_size=_batch_size, shuffle=False)

    # 测试数据集
    _test_dataset = TensorDataset(test_normalize, _test_y)
    _test_loader = DataLoader(_test_dataset, batch_size=_batch_size, shuffle=False)


    # 模型路径以当前脚本所在目录为基准，不受运行时工作目录影响。
    _onnx_path = model_path = Path(__file__).resolve().parent / "models" / "mlp_torch_11f.onnx"
    _onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # train(train_loader=_train_loader, val_loader=_val_loader,
    #       model=_model, loss_fn=_loss_fn, optimizer=_optimizer,
    #       accuracy=_accuracy, auc=_auc, epochs=_epochs)
    #
    # torch.onnx.export(
    #     _model,
    #     (test_normalize[:1],),
    #     _onnx_path,
    #     input_names=[ "features"],
    #     output_names=["logits"],
    #     dynamo=True,
    #     external_data=False
    #
    # )

    session = ort.InferenceSession(str(_onnx_path), providers=["CPUExecutionProvider"])

    for i, (_test, _label) in enumerate(zip(test_normalize[:50], _test_y), start=1):
        onnx_logits = session.run(
            ["logits"],# 要获取名为 logits 的输出 这是模型导出时定义的
            {"features":# 模型输入的名称
                 _test.unsqueeze(0).detach().cpu().numpy()}
        )[0]

        # logits 转为预测概率
        probability = torch.from_numpy(onnx_logits).sigmoid()

        # for probability in probabilities.flatten():
        # _label = True if _label[0] else False
        # print(f"第 i 条：年收入超过 50,000 美元的预测概率为 {probability.item():.2%} {_label}")
        predicted = int(probability.item() >= 0.5)
        actual = int(_label.item())
        result = "正确" if predicted == actual else "错误"

        print(
            f"第 {i} 条：概率 {probability.item():.2%}，"
            f"预测 {predicted}，真实 {actual}，{result}"
        )