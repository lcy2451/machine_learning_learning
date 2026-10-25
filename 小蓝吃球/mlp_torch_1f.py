"""单输入特征、单隐藏层的 PyTorch 神经网络二分类练习。

对应 get_beans4 数据，网络结构为一个输入、两个隐藏神经元和一个输出神经元。
隐藏层与输出层均使用 Sigmoid，通过 MSE 和 SGD 训练，统计训练集指标并绘制预测曲线。
"""
import os
# 本机 OpenMP 重复加载问题的临时绕过设置，并未消除底层运行库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from collections.abc import Callable


import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import torch
from torch.utils.data import TensorDataset, DataLoader
# 使用 TorchMetrics 统计二分类准确率
from torchmetrics.classification import BinaryAccuracy


import dataset
import plot_utils

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
              f" train loss: {epoch_loss:.4f}"
              f" accuracy: {epoch_accuracy:.4f}")


if __name__ == "__main__":

    # 随机种子， 42 是“生命、宇宙以及一切的终极答案”。
    torch.manual_seed(42)

    # 样本数
    m = 100
    # 学习率
    _learning_rate = 0.05
    _batch_size = 10
    _epochs = 10000

    X, Y = dataset.get_beans4(m)

    X = torch.tensor(X, dtype=torch.float).reshape(-1, 1)
    Y = torch.tensor(Y, dtype=torch.float).reshape(-1, 1)

    # 配置图像
    plt.title("Size-Toxicity Function", fontsize=14) #设置图像名称
    plt.xlabel("Beans Size")  # 设置横坐标的名称
    plt.ylabel("Toxicity")  # 设置纵坐标的名称

    plt.scatter(X, Y)

    plt.show()

    _model = torch.nn.Sequential(
        torch.nn.Linear(1, 2),  # 1 个输入 → 2 个隐藏神经元
        torch.nn.Sigmoid(),
        torch.nn.Linear(2, 1),  # 2 个隐藏输出 → 1 个最终输出
        torch.nn.Sigmoid(),
    )

    # 创建优化器
    _optimizer = torch.optim.SGD(_model.parameters(), lr=_learning_rate)

    # 定义指标
    _acc = BinaryAccuracy()

    # 训练数据集
    _dataset = TensorDataset(X, Y)
    _dataloader = DataLoader(_dataset, batch_size=_batch_size, shuffle=True)

    # loss 使用均方误差
    torch.nn.MSELoss()

    train(train_load=_dataloader, model=_model,
          optimizer=_optimizer,
          loss_fn=torch.nn.MSELoss(),
          accuracy=_acc,
          epochs=_epochs)

    _model.eval()
    with torch.no_grad():
        _pres = _model(X).cpu().numpy()

    plt.scatter(X, Y)
    plt.plot(X, _pres)
    plt.show()

