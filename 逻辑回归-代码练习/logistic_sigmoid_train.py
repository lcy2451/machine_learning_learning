from typing import Any
from collections.abc import Callable

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import matplotlib.pyplot as plt

class LogisticRegression(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # 前向传播
        return self.network(x)


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

    # fetch dataset
    banknote_authentication = fetch_ucirepo(id=267)

    # data (as pandas dataframes)
    X = banknote_authentication.data.features
    y = banknote_authentication.data.targets

    # 划分训练集和验证集以及测试集
    x_train, x_temp, y_train, y_temp = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42, shuffle=True)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, train_size=0.5, random_state=42, shuffle=True)

    x_train_tensor = torch.tensor(x_train[["variance", "skewness", "curtosis", "entropy"]].values, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train[["class"]].values, dtype=torch.float)

    x_val_tensor = torch.tensor(x_val[["variance", "skewness", "curtosis", "entropy"]].values, dtype=torch.float)
    y_val_tensor = torch.tensor(y_val[["class"]].values, dtype=torch.float)

    x_test_tensor = torch.tensor(x_test[["variance", "skewness", "curtosis", "entropy"]].values, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test[["class"]].values, dtype=torch.float)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=_batch_size, shuffle=False)

    _loss_fn = torch.nn.BCELoss()

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

    test(
        test_load=test_loader,
        model=_model,
        loss_fn=_loss_fn)

    _model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            prob = _model(X[0])

            print(f"prob {prob[0]:.4f}  y {y[0][0]}")


    plt.xlabel("Training iterations(time)")  # 设置横坐标的名称
    plt.ylabel("Loss")

    # plt.title("l2 0.01")
    plt.plot(_loss_on_trainings, label="Training loss")
    plt.plot(_loss_on_validations, label="Validation loss")
    plt.ylim(0, 0.08)
    # plt.ylim(0, 3500)
    plt.legend()
    plt.show()