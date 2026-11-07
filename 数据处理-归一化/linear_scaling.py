import os
from typing import Any

# 本机 OpenMP 重复加载问题的临时绕过设置，并未消除底层运行库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from collections.abc import Callable

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


# pip install ucimlrepo
from ucimlrepo import fetch_ucirepo

class LinearRegression(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.linear = torch.nn.Linear(in_features=8, out_features=2)

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

    for batch, (batch_x, batch_y) in enumerate(train_load):

        # 1. 前向传播
        pred = model(batch_x)

        # 2. 计算 Loss
        loss = loss_fn(pred, batch_y)

        # 3. 反向传播
        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数

        if batch % 5 == 0:
            loss, current = loss.item(), min((batch + 1) * train_load.batch_size, size)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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
    _batch_size = 64
    # 学习率
    _learning_rate = 0.01


    # fetch dataset
    # https: // archive.ics.uci.edu / dataset / 242 / energy + efficiency
    energy_efficiency = fetch_ucirepo(id=242)

    # data (as pandas dataframes)
    X:pd.DataFrame = energy_efficiency.data.features
    Y:pd.DataFrame  = energy_efficiency.data.targets

    _x = torch.tensor(X[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]].values, dtype=torch.float)
    _y = torch.tensor(Y[["Y1", "Y2"]].values, dtype=torch.float)

    # 创建缩放器，默认范围 [0, 1]
    scaler = MinMaxScaler()

    # 计算每列的最小值、最大值并缩放
    x_scaled = scaler.fit_transform(_x)
    # 转回 Tensor
    x_scaled = torch.as_tensor(x_scaled, dtype=torch.float32)

    x_min = _x.min(dim=0).values
    x_max = _x.max(dim=0).values
    _linear_scaling = (_x- x_min) / (x_max - x_min)


    _x_ds = TensorDataset(_linear_scaling, _y)
    _train_load = DataLoader(
        _x_ds,
        batch_size=_batch_size,
        shuffle=True
    )

    # 这次不测试损失函数， 固定用这个最常用的
    _loss_fn = torch.nn.MSELoss(reduction="mean")

    # 创建模型实例
    _adam_w_model = LinearRegression()


    _adam_w_optimizer = torch.optim.AdamW(
        _adam_w_model.parameters(),  # 需要优化的模型参数
        lr=_learning_rate,  # 学习率
        betas=(0.9, 0.999),  # 一阶与二阶动量的衰减系数
        eps=1e-8,  # 防止除零的小常数
        weight_decay=1e-2  # 权重衰减（AdamW为解耦方式）
    )


    _adam_w_losses = []
    for t in range(_epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(
            train_load=_train_load,
            model=_adam_w_model, optimizer=_adam_w_optimizer,
            loss_fn=_loss_fn)
        test_loss = test(
            test_load=_train_load,
            model=_adam_w_model,
            loss_fn=_loss_fn)

        _adam_w_losses.append(test_loss)
        print("Done!")

    # 按顺序查找可用字体，兼容 macOS 和 Windows
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",  # Windows：微软雅黑
        "PingFang SC",  # macOS：苹方
        "Heiti SC",  # macOS：黑体
        "SimHei",  # Windows：黑体
        "DejaVu Sans",
    ]

    # 训练结束后绘图
    plt.plot(
        range(1, _epochs + 1),
        _adam_w_losses,
        label="AdamW", color="darkviolet",
    )

    plt.xlabel("epochs")  # 设置横坐标的名称
    plt.ylabel("loss")  # 设置纵坐标的名称
    plt.legend()  # 显示 label
    plt.show()
