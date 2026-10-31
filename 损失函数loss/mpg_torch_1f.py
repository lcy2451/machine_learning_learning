import os
# 本机 OpenMP 重复加载问题的临时绕过设置，并未消除底层运行库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from collections.abc import Callable

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


from ucimlrepo import fetch_ucirepo 


def train(
    train_load: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    epochs: int,):

    for epoch in range(epochs):

        model.train()

        for batch_x, batch_y in train_load:

            # 1. 前向传播
            prediction = model(batch_x)

            # 2. 计算 Loss
            loss = loss_fn(prediction, batch_y)

            # 3. 清空梯度
            optimizer.zero_grad()
            
            # 4. 反向传播
            loss.backward()

            # 5. 更新参数
            optimizer.step()
        
        # 把模型切换到评估模式（Evaluation Mode）。
        model.eval()

        # 关闭梯度记录
        with torch.no_grad():
            epoch_loss = 0
            
            for val_x_batch, val_y_batch in train_load:
                # 用验证集验证
                predictions = model(val_x_batch)

                batch_loss = loss_fn(predictions, val_y_batch)
                epoch_loss += batch_loss.item() * len(val_x_batch)

        epoch_loss = epoch_loss / len(train_load.dataset)

        print(f"epoch: {epoch+1}/{epochs}, train loss: {epoch_loss:.4f},")

if __name__ == "__main__":

    # 训练轮数
    _epochs = 5000
    # 批量大小
    _batch_size = 32
    # 学习率
    _learning_rate = 0.001

    # fetch dataset 
    auto_mpg = fetch_ucirepo(id=9) 

    # data (as pandas dataframes) 
    X:pd.DataFrame = auto_mpg.data.features
    Y:pd.DataFrame  = auto_mpg.data.targets

    X["weight"] = X["weight"] / 1000
    _x = torch.tensor(X[["weight"]].values, dtype=torch.float)
    _y = torch.tensor(Y[["mpg"]].values, dtype=float)


    # print(_x, _x_normalize)
    # exit()
    _x_ds = TensorDataset(_x, _y)
    _train_load = DataLoader(
        _x_ds,
        batch_size=_batch_size,
        shuffle=True
    )

    _loss_fn = torch.nn.MSELoss()

    _model = torch.nn.Linear(in_features=1, out_features=1)

    _optimizer = torch.optim.SGD(_model.parameters(), lr=_learning_rate)

    train(
        train_load=_train_load,
        model=_model, optimizer=_optimizer,
        loss_fn=_loss_fn, epochs=_epochs)

    w = _model.weight.item()
    b = _model.bias.item()

    print(f"w = {w:.4f}\n b = {b:.4f}")

    y_pre = b + (w*_x)

    plt.plot(_x, y_pre, color="red")
    plt.scatter(_x, _y, s=10, alpha=0.3)
    plt.show()
