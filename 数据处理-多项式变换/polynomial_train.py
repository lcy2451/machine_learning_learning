from typing import Any
from collections.abc import Callable

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import MinMaxScaler

from ucimlrepo import fetch_ucirepo 

class LinearRegression(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.linear = torch.nn.Linear(in_features=5, out_features=1)

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
    _batch_size = 32
    # 学习率
    _learning_rate = 0.01

    # fetch dataset 
    auto_mpg = fetch_ucirepo(id=9) 
    
    # data (as pandas dataframes) 
    X:pd.DataFrame = auto_mpg.data.features.copy()
    Y:pd.DataFrame  = auto_mpg.data.targets

    # 发现horsepower字段有6个Nan值， 给删了， 本就是小数据集， 更小了
    X = X.dropna(how="any", subset=["horsepower"])
    # 不能只删除X
    Y = Y.loc[X.index]

    X["horsepower2"] = X["horsepower"] **2
    X["horsepower3"] = X["horsepower"] **3
    X["horsepower4"] = X["horsepower"] **4
    X["horsepower5"] = X["horsepower"] **5

    # 创建缩放器，默认范围 [0, 1]
    xscaler = MinMaxScaler()
    yscaler = MinMaxScaler()


    # 计算每列的最小值、最大值并缩放
    # 归一化
    scaled_x = xscaler.fit_transform(X[
        ["horsepower", "horsepower2", "horsepower3", "horsepower4", "horsepower5"]
        ])
    # 转回 Tensor
    scaled_x = torch.as_tensor(scaled_x, dtype=torch.float32)

    scaled_y = yscaler.fit_transform(Y[["mpg"]])
    scaled_y = torch.as_tensor(scaled_y, dtype=torch.float32)


    _train_dataset = TensorDataset(scaled_x, scaled_y)
    _train_dataload = DataLoader(_train_dataset, batch_size=_batch_size, shuffle=True)

    # 损失函数 模型 优化器
    _loss_fn = torch.nn.MSELoss(reduction="mean")
    _model = LinearRegression()
    _optimizer = torch.optim.SGD(_model.parameters(), lr=_learning_rate)

    for t in range(_epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(
                train_load=_train_dataload,
                model=_model, optimizer=_optimizer,
                loss_fn=_loss_fn)
            test(
                test_load=_train_dataload,
                model=_model,
                loss_fn=_loss_fn)
    
            print("Done!")

    # 按顺序查找可用字体，兼容 macOS 和 Windows
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",  # Windows：微软雅黑
        "PingFang SC",      # macOS：苹方
        "Heiti SC",         # macOS：黑体
        "SimHei",          # Windows：黑体
        "DejaVu Sans",
    ]

    w1, w2, w3, w4, w5  = _model.linear.weight[0].tolist()
    b = _model.linear.bias.item()

    y_pre = b + (w1*scaled_x[:, 0]) + (w2*scaled_x[:, 1]) + (w3*scaled_x[:, 2]) + (w4*scaled_x[:, 3]) + (w5*scaled_x[:, 4])

    sorted_indices = torch.argsort(scaled_x[:, 0])
    plt.plot(
    scaled_x[sorted_indices, 0],
    y_pre[sorted_indices],
    color="red"
    )

    plt.scatter(scaled_x[:, 0], scaled_y[:, 0], s=10, alpha=1)

    plt.xlabel("马力")  # 设置横坐标的名称
    plt.ylabel("燃油经济性（MPG）")  # 设置纵坐标的名称
    plt.show()