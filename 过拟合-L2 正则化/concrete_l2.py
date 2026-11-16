from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class LinearRegression(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        # self.linear = torch.nn.Linear(in_features=8, out_features=1)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(8, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
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

def normalize(x) -> MinMaxScaler:
    # 创建缩放器，默认范围 [0, 1]
    scaler = MinMaxScaler()
    scaler.fit(x)
    return scaler


if __name__ == "__main__":
    torch.manual_seed(42)

    # 训练轮数
    _epochs = 500
    # 批量大小
    _batch_size = 64
    # 学习率
    _learning_rate = 0.01
    _momentum = 0.9

    # fetch dataset
    concrete_compressive_strength = fetch_ucirepo(id=165)

    # data (as pandas dataframes)
    X = concrete_compressive_strength.data.features
    y = concrete_compressive_strength.data.targets

    # print(y[["Concrete compressive strength"]])

    x_train, x_temp, y_train, y_temp = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42, shuffle=True)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, train_size=0.5, random_state=42, shuffle=True)

    # 归一化
    # 计算每列的最小值、最大值并缩放
    _scaler = normalize(x_train[[
        "Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer", "Coarse Aggregate",
        "Fine Aggregate", "Age"]])
    _train_input_transform = _scaler.transform(x_train[[
        "Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer", "Coarse Aggregate",
        "Fine Aggregate", "Age"]])
    _val_input_transform = _scaler.transform(x_val[[
        "Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer", "Coarse Aggregate",
        "Fine Aggregate", "Age"]])
    _test_input_transform = _scaler.transform(x_test[[
        "Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer", "Coarse Aggregate",
        "Fine Aggregate", "Age"]])

    _y_scaler = normalize(y_train[["Concrete compressive strength"]])
    _train_target_transform = _y_scaler.transform(y_train[["Concrete compressive strength"]])
    _val_target_transform = _y_scaler.transform(y_val[["Concrete compressive strength"]])
    _test_target_transform = _y_scaler.transform(y_test[["Concrete compressive strength"]])

    # 转回 Tensor
    _train_tensor = torch.as_tensor(_train_input_transform, dtype=torch.float32)
    _train_target_tensor = torch.as_tensor(_train_target_transform, dtype=torch.float32)

    _val_tensor = torch.as_tensor(_val_input_transform, dtype=torch.float32)
    _val_target_tensor = torch.as_tensor(_val_target_transform, dtype=torch.float32)

    _test_tensor = torch.as_tensor(_test_input_transform, dtype=torch.float32)
    _test_target_tensor = torch.as_tensor(_test_target_transform, dtype=torch.float32)

    _train_dataset = TensorDataset(_train_tensor, _train_target_tensor)
    _train_dataloader = DataLoader(_train_dataset, batch_size=42, shuffle=True)

    _val_dataset = TensorDataset(_val_tensor, _val_target_tensor)
    _val_dataloader = DataLoader(_val_dataset, batch_size=42, shuffle=False)

    _test_dataset = TensorDataset(_test_tensor, _test_target_tensor)
    _test_dataloader = DataLoader(_test_dataset, batch_size=42, shuffle=False)

    # 损失函数 模型 优化器
    _loss_fn = torch.nn.MSELoss(reduction="mean")
    _model = LinearRegression()
    _optimizer = torch.optim.SGD(_model.parameters(), lr=_learning_rate, weight_decay=0.0001)

    _loss_on_validations = []
    _loss_on_trainings = []
    for t in range(_epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss = train(
            train_load=_train_dataloader,
            model=_model, optimizer=_optimizer,
            loss_fn=_loss_fn)
        _loss_on_trainings.append(train_loss)

        test_loss = test(
            test_load=_val_dataloader,
            model=_model,
            loss_fn=_loss_fn)
        _loss_on_validations.append(test_loss)

        print("Done!")

    test(
        test_load=_test_dataloader,
        model=_model,
        loss_fn=_loss_fn)

    plt.xlabel("Training iterations(time)")  # 设置横坐标的名称
    plt.ylabel("Loss")

    plt.title("l2 0.0001")
    plt.plot(_loss_on_trainings, label="Training loss")
    plt.plot(_loss_on_validations, label="Validation loss")
    plt.ylim(0, 0.15)
    plt.legend()
    plt.show()
