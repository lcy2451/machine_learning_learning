from typing import Any
from collections.abc import Callable

import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import BinaryConfusionMatrix
import matplotlib.pyplot as plt
# 安装seaborn库
import seaborn as sns

class LogisticRegression(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.linear = torch.nn.Linear(in_features=30, out_features=1)

    def forward(self, x):
        # 前向传播
        return self.linear(x)


def train(
    train_load: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):

    size = len(train_load.dataset)
    model.train()

    test_loss = 0

    for batch, (batch_x, batch_y) in enumerate(train_load):

        # 1. 前向传播
        pred = model(batch_x)

        # 2. 计算 Loss
        loss = loss_fn(pred, batch_y)

        # 3. 反向传播
        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数

        if batch % 5 == 0 or True:
            loss, current = loss.item(), min((batch + 1) * train_load.batch_size, size)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            test_loss += loss
    test_loss /= len(train_load)
    return test_loss


def test(
    test_load: DataLoader,
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
):
    num_batches = len(test_load)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in test_load:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n , Avg loss: {test_loss:>8f} \n")

    return test_loss


if __name__ == "__main__":
    # 训练轮数
    _epochs = 500
    # 批量大小
    _batch_size = 32
    # 学习率
    _learning_rate = 0.01

    # 阈值
    _threshold = 0.5

    data:sklearn.utils._bunch.Bunch = load_breast_cancer()
    X = data.data
    Y = data.target

    # 划分训练集和验证集以及测试集
    x_train, x_temp, y_train, y_temp = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=42, shuffle=True)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, train_size=0.5, random_state=42, shuffle=True)

    # 归一化
    # 创建缩放器，默认范围
    # [0, 1]
    x_scaler = MinMaxScaler()
    # 计算每列的最小值、最大值并缩放
    x_scaler.fit(x_train)
    x_train_scaled = x_scaler.transform(x_train)
    x_val_scaled = x_scaler.transform(x_val)
    x_test_scaled = x_scaler.transform(x_test)

    # 归一化
    # 创建缩放器，默认范围
    # [0, 1]
    y_scaler = MinMaxScaler()
    # 计算每列的最小值、最大值并缩放
    y_scaler.fit(y_train.reshape(-1, 1))
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1))
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

    _x_train_tensor = torch.as_tensor(x_train_scaled, dtype=torch.float)
    _y_train_tensor = torch.as_tensor(y_train_scaled, dtype=torch.float)
    _x_val_tensor = torch.as_tensor(x_val_scaled, dtype=torch.float)
    _y_val_tensor = torch.as_tensor(y_val_scaled, dtype=torch.float)
    _x_test_tensor = torch.as_tensor(x_test_scaled, dtype=torch.float)
    _y_test_tensor = torch.as_tensor(y_test_scaled, dtype=torch.float)

    train_dataset = TensorDataset(_x_train_tensor, _y_train_tensor)
    val_dataset = TensorDataset(_x_val_tensor, _y_val_tensor)
    test_dataset = TensorDataset(_x_test_tensor, _y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=_batch_size, shuffle=False)

    _loss_fn = torch.nn.BCEWithLogitsLoss()

    # 创建模型实例
    _model = LogisticRegression()

    # 优化器
    _optimizer = torch.optim.AdamW(
        _model.parameters(),  # 需要优化的模型参数
        lr=_learning_rate,  # 学习率
        betas=(0.9, 0.999),  # 一阶与二阶动量的衰减系数
        eps=1e-8,  # 防止除零的小常数
        weight_decay=1e-2  # 权重衰减（AdamW为解耦方式）
    )

    _loss_on_validations = []
    _loss_on_trainings = []
    for t in range(_epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss = train(
            train_load=train_loader,
            model=_model, optimizer=_optimizer,
            loss_fn=_loss_fn)
        _loss_on_trainings.append(train_loss)

        test_loss = test(
            test_load=val_loader,
            model=_model,
            loss_fn=_loss_fn)
        _loss_on_validations.append(test_loss)

        print("Done!")
    print("end")

    _model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            logits = _model(X[0])

            # 使用 sigmoid 把它变成 0~1 的概率
            prob = torch.sigmoid(logits)
            pred = torch.where(prob > _threshold, 1, 0)
            print(f"pred {pred[0]}  y {y[0][0]}")
        for X, y in test_loader:
            logits = _model(X)
            # 使用 sigmoid 把它变成 0~1 的概率
            predictions = torch.sigmoid(logits)
            confusion_matrix = BinaryConfusionMatrix(threshold=0.5)
            result = confusion_matrix(predictions, y)
            print(result)

            matrix = result.cpu().numpy()
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"]
            )

            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.show()
