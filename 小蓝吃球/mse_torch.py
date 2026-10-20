"""教程第二节：使用 PyTorch 练习均方误差（MSE）损失函数。"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import dataset
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader


def train(train_loader:DataLoader, model:torch.nn.Module, loss_fn, epochs:int, learning_rate=0.05 ):

    # 创建优化器 Adam
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


    # 模型切换到训练
    model.train()
    for epoch in range(epochs):

        for x_batch, y_batch in train_loader:

            # 前向传播
            _logits = model(x_batch)

            # 计算 Loss
            loss = loss_fn(_logits, y_batch)

            # 清空梯度
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()


        print(f"第 {epoch + 1} 轮, MSE：{loss.item():.6f}")


if __name__ == "__main__":
    # 随机种子， 42 是“生命、宇宙以及一切的终极答案”。
    torch.manual_seed(42)

    _batch_size = 50
    _epochs = 100
    _learning_rate = 0.005

    xs, ys = dataset.get_beans(100)

    x = torch.tensor(xs, dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(ys, dtype=torch.float32).reshape(-1, 1)

    _mean = torch.mean(x)
    _std = x.std(dim=0, unbiased=False).clamp(min=1e-8)
    _normalize = (x - _mean) / _std

    x_dataset = TensorDataset(_normalize, y)
    x_dataloader = DataLoader(x_dataset, batch_size=_batch_size)

    _model = torch.nn.Linear(in_features=1, out_features=1)

    # torch.nn.MSELoss() 本身是一个“损失函数模块 / 损失函数形式”；当它把一批或整个训练集的误差聚合成一个标量、作为优化器要最小化的目标时，这个标量目标可以叫“代价函数”。
    _loss_fn = torch.nn.MSELoss()
    train(train_loader=x_dataloader, model=_model, loss_fn=_loss_fn, epochs=_epochs, learning_rate=_learning_rate)

    _w_1 = _model.weight.item()
    _b = _model.bias.item()
    _x_1 = _normalize
    print(f"标准化输入的权重：{_w_1}")
    print(f"标准化输入的截距：{_b}")

    #  下面是线性回归计算， y=b+wx
    y_pre = _b + _x_1 * _w_1

    # 配置图像
    plt.title("Size-Toxicity Function", fontsize=14)  # 设置图像名称
    plt.xlabel("Beans Size")  # 设置横坐标的名称
    plt.ylabel("Toxicity")  # 设置纵坐标的名称

    # w = 1.2633912772426041
    # y_pre = w * xs

    plt.plot(xs, y_pre)
    plt.scatter(xs, ys)
    plt.show()
