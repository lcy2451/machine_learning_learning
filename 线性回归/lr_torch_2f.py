import os
# 本机 OpenMP 重复加载问题的临时绕过设置，并未消除底层运行库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.regression import MeanAbsoluteError

from train_utils import train


if __name__ == '__main__':
    # 随机种子， 42 是“生命、宇宙以及一切的终极答案”。
    torch.manual_seed(42)

    # @title
    _csv =  Path("data/chicago_taxi_train.csv")
    if not _csv.exists():
        _csv = Path("../data/chicago_taxi_train.csv")
    chicago_taxi_dataset = pd.read_csv(_csv)
    # TRIP_MILES  行程里程
    # TRIP_SECONDS 行程时长（秒）
    # FARE 票价
    # COMPANY  公司
    # PAYMENT_TYPE  支付方式
    # TIP_RATE 小费
    _training_df:pd.Series | pd.DataFrame = chicago_taxi_dataset.loc[
        :, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]


    _training_df["TRIP_MINUTES"] = _training_df["TRIP_SECONDS"] / 60

    # 把 Pandas 数据转换成 PyTorch 的 Tensor（张量）。
    # 输入特征 X：行程里程
    _x:torch.Tensor = torch.tensor(
        _training_df[['TRIP_MILES', 'TRIP_MINUTES']].values, dtype
        =torch.float)
    # 真实标签 y：车费
    _y:torch.Tensor = torch.tensor(_training_df[['FARE']].values, dtype=torch.float)


    _epochs = 50
    _batch_size = 50
    # 学习率
    _learning_rate = 0.001

    mean = _x.mean(dim=0)
    std = _x.std(dim=0, unbiased=False)

    # 防止标准差为 0
    std = std.clamp(min=1e-7)
    _x = (_x - mean) / std

    # 训练数据集
    dataset = TensorDataset(_x, _y)
    _train_loader = DataLoader(
    dataset,
    batch_size=_batch_size,
    shuffle=True
    )

    # L1 loss
    # _loss_fn = torch.nn.L1Loss(reduction="sum")
    _loss_fn = torch.nn.MSELoss()
    _loss_fn = torch.nn.L1Loss(reduction="sum")

    # in_features = 2  输入特征数量
    # out_features = 1 输出特征数量
    _model = torch.nn.Linear(in_features=2, out_features=1)

    # 优化器
    _optimizer = torch.optim.RMSprop(_model.parameters(), lr=_learning_rate )

    # 定义指标
    _accuracy = MeanAbsoluteError()
    train(
        train_load=_train_loader,
        model=_model,
        optimizer=_optimizer,
        loss_fn=_loss_fn,
        accuracy=_accuracy,
        epochs=_epochs)

    # 按顺序查找可用字体，兼容 macOS 和 Windows
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",  # Windows：微软雅黑
        "PingFang SC",  # macOS：苹方
        "Heiti SC",  # macOS：黑体
        "SimHei",  # Windows：黑体
        "DejaVu Sans",
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # 设置 Y 轴朝上
    ax.view_init(elev=25, azim=-60, vertical_axis="y")
    # ax = Axes3D(fig)

    # 随机抽取 2000 个样本
    sample_count = min(2000, len(_x))
    indices = torch.randperm(len(_x))[:sample_count]



    x = _x[indices, 0].detach().cpu().numpy()
    z = _x[indices, 1].detach().cpu().numpy()
    y = _y[indices].flatten().detach().cpu().numpy()

    # ax.set_xlim(-2, 7)
    # ax.set_ylim(-2, 150)
    # ax.set_zlim(-2, 7)
    ax.scatter(x, y, z, alpha=0.5)

    # 创建二维坐标网格
    x_range = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 30)
    z_range = np.linspace(np.percentile(z, 1), np.percentile(z, 99), 30)

    X, Z = np.meshgrid(x_range, z_range)

    # 将网格转换成模型需要的 (N, 2)
    grid = np.column_stack([X.ravel(), Z.ravel()])
    grid_tensor = torch.from_numpy(grid).float()

    # 对网格上的每一个点进行预测
    _model.eval()
    with torch.no_grad():
        Y = _model(grid_tensor).numpy().reshape(X.shape)

    ax.plot_surface(X, Y, Z, cmap="rainbow", alpha=1)

    plt.show()


