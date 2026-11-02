import os
from typing import Any

# 本机 OpenMP 重复加载问题的临时绕过设置，并未消除底层运行库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from collections.abc import Callable

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from ucimlrepo import fetch_ucirepo 

class LinearRegression(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.linear = torch.nn.Linear(in_features=1, out_features=1)

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


if __name__ == "__main__":

    # 训练轮数
    _epochs = 2000
    # 批量大小
    _batch_size = 32
    # 学习率
    _learning_rate = 0.01

    # fetch dataset 
    auto_mpg = fetch_ucirepo(id=9) 

    # data (as pandas dataframes) 
    X:pd.DataFrame = auto_mpg.data.features
    Y:pd.DataFrame  = auto_mpg.data.targets

    X["weight"] = X["weight"] / 1000
    _x = torch.tensor(X[["weight"]].values, dtype=torch.float)
    _y = torch.tensor(Y[["mpg"]].values, dtype=torch.float)


    _x_ds = TensorDataset(_x, _y)
    _train_load = DataLoader(
        _x_ds,
        batch_size=_batch_size,
        shuffle=True
    )

    # _loss_fn = torch.nn.MSELoss(reduction="mean")
    # _loss_fn = torch.nn.HuberLoss(reduction="mean")
    # _loss_fn = torch.nn.L1Loss(reduction="mean")
    _loss_fn = torch.nn.SmoothL1Loss(reduction="mean")

    # 创建模型实例
    _model = LinearRegression()

    _optimizer = torch.optim.SGD(_model.parameters(), lr=_learning_rate)

    for t in range(_epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(
            train_load=_train_load,
            model=_model, optimizer=_optimizer,
            loss_fn=_loss_fn)
        test(
            test_load=_train_load,
            model=_model,
            loss_fn=_loss_fn)
        print("Done!")

    w = _model.linear.weight.item()
    b = _model.linear.bias.item()

    print(f"w = {w:.4f}\n b = {b:.4f}")

    y_pre = b + (w*_x)

    # 按顺序查找可用字体，兼容 macOS 和 Windows
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",  # Windows：微软雅黑
        "PingFang SC",      # macOS：苹方
        "Heiti SC",         # macOS：黑体
        "SimHei",          # Windows：黑体
        "DejaVu Sans",
    ]


    plt.plot(_x, y_pre, color="red")
    plt.scatter(_x, _y, s=10, alpha=0.3)

    plt.xlabel("重量")  # 设置横坐标的名称
    plt.ylabel("燃油经济性（MPG）")  # 设置纵坐标的名称
    plt.show()
