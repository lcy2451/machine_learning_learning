"""双输入特征的 PyTorch 螺旋二分类练习。

使用 get_beans6 生成两类带坐标噪声的螺旋（蚊香）数据，两个输入特征为平面坐标。
网络结构为 2 → 8 → 8 → 8 → 1，三个隐藏层使用 ReLU，输出层使用 Sigmoid。
通过 PyTorch 自动求导，以 MSE 损失和 SGD 优化器分批训练，每轮评估训练集指标。
训练后在二维网格上预测并绘制三维曲面，尚未划分独立验证集或测试集。
"""
import os
# 本机 OpenMP 重复加载问题的临时绕过设置，并未消除底层运行库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from collections.abc import Callable

import matplotlib.pyplot as plt
import torch
# 使用 TorchMetrics 统计二分类准确率
from torchmetrics.classification import BinaryAccuracy
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import dataset
import plot_utils

def show_scatter_surface_with_model(X, Y, model):
    """接收 NumPy 样本，生成二维网格并使用 PyTorch 模型绘制预测曲面。"""

    x = X[:, 0]
    z = X[:, 1]
    y = Y

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # ax = Axes3D(fig)
    ax.scatter(x, z, y)

    x = np.arange(np.min(x), np.max(x), 0.1)
    z = np.arange(np.min(z), np.max(z), 0.1)
    x, z = np.meshgrid(x, z)

    X = np.column_stack((x[0], z[0]))

    for j in range(z.shape[0]):
        if j == 0:
            continue
        X = np.vstack((X, np.column_stack((x[0], z[j]))))

    y = model(torch.tensor(X, dtype=torch.float))

    # 解除梯度关联并转成 NumPy 数组，再恢复为曲面网格形状
    y = np.array([y.detach().cpu().numpy()])
    y = y.reshape(x.shape[0], z.shape[1])
    ax.plot_surface(x, z, y, cmap='rainbow')
    plt.show()

def train(train_load:DataLoader, model:torch.nn.Module,
          optimizer:torch.optim.Optimizer,
          loss_fn:Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
          accuracy: BinaryAccuracy,
          epochs:int):

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

        # 每个 Epoch 开始评估时重置指标
        accuracy.reset()

        # 关闭梯度记录
        with torch.no_grad():

            epoch_loss = 0
            for val_x_batch, val_y_batch in train_load:
                # 在训练集上评估，不是独立验证集
                predictions = model(val_x_batch)

                batch_loss = loss_fn(predictions, val_y_batch)
                epoch_loss += batch_loss * len(val_x_batch)

                # 更新指标
                probabilities = predictions.flatten()
                targets = val_y_batch.int().flatten()
                accuracy.update(probabilities, targets)

        epoch_loss = epoch_loss / len(train_load.dataset)

        epoch_accuracy = accuracy.compute()
        print(f"epoch: {epoch + 1}/{epochs}"
              f" train loss: {epoch_loss:.8f}"
              f" accuracy: {epoch_accuracy:.4f}")

if __name__ == "__main__":

    # 固定 PyTorch 随机性；螺旋数据的坐标噪声使用 Python random，未在此固定
    torch.manual_seed(42)

    # 样本数
    m = 100
    # 学习率
    _learning_rate = 0.03
    _batch_size = 10
    _epochs = 50000

    _X, _Y = dataset.get_beans6(m)
    # plot_utils.show_scatter(_X, _Y)

    # 两个坐标保留 (m, 2)，标签调整为 (m, 1) 以匹配模型输出
    X = torch.tensor(_X, dtype=torch.float)
    Y = torch.tensor(_Y, dtype=torch.float).reshape(-1, 1)

    # print(_X.shape, _Y.shape)
    # print(_Y)

    _model = torch.nn.Sequential(
        torch.nn.Linear(2, 8),
        torch.nn.ReLU(),

        torch.nn.Linear(8, 8),
        torch.nn.ReLU(),

        torch.nn.Linear(8, 8),
        torch.nn.ReLU(),

        torch.nn.Linear(8, 1),
        torch.nn.Sigmoid(),
    )

    # 创建优化器
    _optimizer = torch.optim.SGD(_model.parameters(), lr=_learning_rate)

    # 定义指标
    _acc = BinaryAccuracy()

    # loss 使用均方误差
    torch.nn.MSELoss()

    # 训练数据集
    _dataset = TensorDataset(X, Y)
    _dataloader = DataLoader(_dataset, batch_size=_batch_size, shuffle=True)

    train(train_load=_dataloader, model=_model,
          optimizer=_optimizer,
          loss_fn=torch.nn.MSELoss(),
          accuracy=_acc,
          epochs=_epochs)

    _model.eval()
    with torch.no_grad():
        _pres = _model(X).cpu().numpy()

    show_scatter_surface_with_model(_X, _Y, _model)
