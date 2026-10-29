# 使用 芝加哥出租车数据，学习行程里程与车费之间的线性关系。
# pandas 负责读取数据，PyTorch 负责训练，Matplotlib 负责绘制结果。

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.regression import MeanAbsoluteError

from train_utils import train


if __name__ == '__main__':
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    # 可以选择随机抽样， 用一部分数据来训练， 这里我选择用全部的数据
    # 可选
    # training_df = training_df.sample(
    #     n=min(200, len(training_df)),
    #     random_state=42
    # )

    # 把 Pandas 数据转换成 PyTorch 的 Tensor（张量）。
    # 输入特征 X：行程里程
    _x:torch.Tensor = torch.tensor(_training_df[['TRIP_MILES']].values, dtype=torch.float)
    # 真实标签 y：车费
    _y:torch.Tensor = torch.tensor(_training_df[['FARE']].values, dtype=torch.float)

    # 训练轮数
    _epochs = 20
    # 批量大小
    _batch_size = 50
    # 学习率
    _learning_rate = 0.001

    # L1 loss
    _loss_fn = torch.nn.L1Loss(reduction="sum")

    # 训练数据集
    _dataset = TensorDataset(_x, _y)

    # 创建数据加载器
    # 假设你有 200 条训练数据，而模型每次只需要处理 50 条。
    # DataLoader 就可以把数据分成 4 个 Batch（批次），让训练循环依次获取。
    _train_loader = DataLoader(
        _dataset,
        batch_size=_batch_size,
        shuffle=True
    )

    _model = torch.nn.Linear(in_features=1, out_features=1)

    # 优化器：RMSprop
    _optimizer = torch.optim.RMSprop(_model.parameters(), lr=_learning_rate)

    # 定义指标
    _accuracy = MeanAbsoluteError().to(_device)

    train(
        train_load=_train_loader,
        model=_model,
        optimizer=_optimizer,
        loss_fn=_loss_fn,
        accuracy=_accuracy,
        epochs=_epochs)

    weight = _model.weight.item()
    bais = _model.bias.item()
    #  下面是线性回归计算， y=b+wx
    y_pre = bais + _x * weight

    # 按顺序查找可用字体，兼容 macOS 和 Windows
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",  # Windows：微软雅黑
        "PingFang SC",      # macOS：苹方
        "Heiti SC",         # macOS：黑体
        "SimHei",          # Windows：黑体
        "DejaVu Sans",
    ]

    # 避免部分中文字体无法显示负号
    plt.rcParams["axes.unicode_minus"] = False
        # 配置图像
    plt.title("行驶里程与车费", fontsize=14)  # 设置图像名称
    plt.xlabel("行程里程")  # 设置横坐标的名称
    plt.ylabel("车费")  # 设置纵坐标的名称

    # w = 1.2633912772426041
    # y_pre = w * xs

    plt.plot(_x, y_pre, color="red")
    plt.scatter(_x, _y, s=10, alpha=0.3)
    plt.show()

