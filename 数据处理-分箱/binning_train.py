from typing import Any
from collections.abc import Callable

import matplotlib.pyplot as plt

import pandas as pd
# 使用pip安装sklearn 用来分训练集， 验证集， 测试集 pip install -U scikit-learn
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

from ucimlrepo import fetch_ucirepo 

class LinearRegression(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.linear = torch.nn.Linear(in_features=2, out_features=1)

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
    _momentum = 0.9

    # fetch dataset 
    iris = fetch_ucirepo(id=53) 
    
    # data (as pandas dataframes) 
    X:pd.DataFrame = iris.data.features 
    Y:pd.DataFrame = iris.data.targets 

    # 划分 80训练集. 80标签
    _train_df, _temp_df, _label_df, _label_temp_df = train_test_split(
        X[["petal length"]], X[["petal width"]], test_size=0.2, random_state=42)

    # 划分验证集和测试集. 划分标签验证集和标签测试集
    _val_df, _test_df, _label_val_df, _label_test_df = train_test_split(
        _temp_df, _label_temp_df, test_size=0.5, random_state=42
    )

    # 分箱
    bins = [0, 2.5, 8]
    _labels = ["短花瓣", "长花瓣"]
    categories = pd.cut(_train_df["petal length"], bins, labels=_labels)
    val_categories = pd.cut(_val_df["petal length"], bins=bins, labels=_labels)
    test_categories = pd.cut(_test_df["petal length"], bins=bins, labels=_labels)

    # 每个箱的样本数量
    # print(categories.value_counts())
    # print(categories.dtype) # category

    # 独热编码 One-Hot
    encoded = pd.get_dummies(categories, dtype=int)
    val_encoded = pd.get_dummies(val_categories, dtype=int)
    test_encoded = pd.get_dummies(test_categories, dtype=int)

    print(test_encoded)
    exit()

    # to tensor
    _x = torch.tensor(encoded.to_numpy(), dtype=torch.float32)
    _val = torch.tensor(val_encoded.to_numpy(), dtype=torch.float32)
    _test = torch.tensor(test_encoded.to_numpy(), dtype=torch.float32)

    _label = torch.tensor(_label_df.to_numpy(), dtype=torch.float32)
    _val_label = torch.tensor(_label_val_df.to_numpy(), dtype=torch.float32)
    _test_label = torch.tensor(_label_test_df.to_numpy(), dtype=torch.float32)

    # to TensorDataset
    _train_dataset = TensorDataset(_x, _label)
    _val_dataset = TensorDataset(_val, _val_label)
    _test_dataset = TensorDataset(_test, _test_label)

    # to data load
    _train_dataload = DataLoader(_train_dataset, batch_size=_batch_size, shuffle=True)
    _val_dataload = DataLoader(_val_dataset, batch_size=_batch_size, shuffle=False)
    _test_dataload = DataLoader(_test_dataset, batch_size=_batch_size, shuffle=False)

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
            test_load=_val_dataload,
            model=_model,
            loss_fn=_loss_fn)

        print("Done!")

    test(
        test_load=_test_dataload,
        model=_model,
        loss_fn=_loss_fn)
    
    


    # 按顺序查找可用字体，兼容 macOS 和 Windows
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",  # Windows：微软雅黑
        "PingFang SC",      # macOS：苹方
        "Heiti SC",         # macOS：黑体
        "SimHei",          # Windows：黑体
        "DejaVu Sans",
    ]
    plt.xlabel("petal length 花瓣长度")
    plt.ylabel("petal width 花瓣宽度")
    plt.scatter(
        _train_df["petal length"],
        _label_df["petal width"]
        )
    
    plt.axvline(
        x=2.5,
        color="red",
        linestyle="--",
        alpha=0.7
    )
    plt.show()
