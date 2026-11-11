from typing import Any
from collections.abc import Callable


# Bike Sharing 数据集：对 season 进行独热编码，预测 cnt。
from ucimlrepo import fetch_ucirepo 
import pandas as pd
# 使用pip安装sklearn 用来分训练集， 验证集， 测试集 pip install -U scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class LinearRegression(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        # self.linear = torch.nn.Linear(in_features=10, out_features=1)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=4+1+7+4+12+24+7, out_features=256), 
            torch.nn.ReLU(),

            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),

            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),

            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),

            torch.nn.Linear(64, 1),
        )

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
    _batch_size = 200
    # 学习率
    _learning_rate = 0.001

    # fetch dataset 
    bike_sharing = fetch_ucirepo(id=275) 
    
    # data (as pandas dataframes) 
    X = bike_sharing.data.features 
    y = bike_sharing.data.targets 

    # 划分 80训练集
    _train_df, _temp_df = train_test_split(
        X, test_size=0.2, random_state=42)
    _trai_val_df, _trai_test_df = train_test_split(
            _temp_df, test_size=0.5, random_state=42)
    
    _label_df, _temp_df = train_test_split(
        y, test_size=0.2, random_state=42)
    
    _label_val_df, _label_test_df = train_test_split(
        _temp_df, test_size=0.5, random_state=42)

    # 创建一个包含分类变量的 Series
    mnth_encoder = OneHotEncoder()
    # 学习规则
    mnth_encoder.fit(_train_df[["mnth"]])
    mnth_transformed = mnth_encoder.transform(_train_df[["mnth"]])
    mnth_val_transformed = mnth_encoder.transform(_trai_val_df[["mnth"]])

    # #  创建一个包含分类变量的 Series
    hr_encoder = OneHotEncoder()
    # 学习规则
    hr_encoder.fit(_train_df[["hr"]])
    hr_transformed = hr_encoder.transform(_train_df[["hr"]])
    hr_val_transformed = hr_encoder.transform(_trai_val_df[["hr"]])
    # print(len(hr_encoder.categories_[0]))
    # print(y.columns)
    
    dates = pd.to_datetime(_train_df["dteday"])
    elapsed_days = (dates - dates.min()).dt.days

    # 归一化
    # 创建缩放器，默认范围 [0, 1]
    scaler = MinMaxScaler()
    # 计算每列的最小值、最大值并缩放
    scaler.fit(elapsed_days.to_frame())
    dteday_scaled = scaler.transform(elapsed_days.to_frame())

    train_dates = pd.to_datetime(_trai_val_df["dteday"])
    elapsed_days = (train_dates - dates.min()).dt.days
    dteday_val_scaled = scaler.transform(elapsed_days.to_frame())

    # 初始化 OneHotEncoder 对象
    encoder = OneHotEncoder()
    # X["season"]:pd.Series
    # 学习规则
    encoder.fit(_train_df[["season"]])

    # encoder.transform()
    # print(encoder.categories_)
    # 转换，不重新学习规则
    season_transformed = encoder.transform(_train_df[["season"]])
    season_val_transformed = encoder.transform(_trai_val_df[["season"]])

    weekday_encoder = OneHotEncoder()
    # 学习规则
    weekday_encoder.fit(_train_df[["weekday"]])
    # 转换，不重新学习规则
    weekday_transformed = weekday_encoder.transform(_train_df[["weekday"]])
    weekday_val_transformed = weekday_encoder.transform(_trai_val_df[["weekday"]])

    weathersit_encoder = OneHotEncoder()
    # 学习规则
    weathersit_encoder.fit(_train_df[["weathersit"]])
    # 转换，不重新学习规则
    weathersit_transformed = weathersit_encoder.transform(_train_df[["weathersit"]])
    weathersit_val_transformed = weathersit_encoder.transform(_trai_val_df[["weathersit"]])

    combined = np.concatenate(
    [
        season_transformed.toarray(), # 4
        dteday_scaled, # 1
        weekday_transformed.toarray(), # 7
        weathersit_transformed.toarray(), # 4
        mnth_transformed.toarray() # 12
        , hr_transformed.toarray(), #24
       _train_df[["yr", "temp", "hum", "atemp", "holiday", "workingday", "windspeed"]].values #7
       ],
    axis=1
    )

    _x = torch.as_tensor(combined, dtype=torch.float32)
    _y = torch.tensor(_label_df[["cnt"]].values, dtype=torch.float32)

    val_combined = np.concatenate(
    [
        season_val_transformed.toarray(), # 4
        dteday_val_scaled, # 1
        weekday_val_transformed.toarray(), # 7
        weathersit_val_transformed.toarray(), # 4
        mnth_val_transformed.toarray() # 12
        , hr_val_transformed.toarray(), #24
       _trai_val_df[["yr", "temp", "hum", "atemp", "holiday", "workingday", "windspeed"]].values #7
       ],
    axis=1
    )
    _val_x = torch.as_tensor(val_combined, dtype=torch.float32)
    _val_y = torch.tensor(_label_val_df[["cnt"]].values, dtype=torch.float32)


    _train_dataset = TensorDataset(_x, _y)
    _train_dataload = DataLoader(_train_dataset, batch_size=_batch_size, shuffle=True)

    _val_dataset = TensorDataset(_val_x, _val_y)
    _val_dataload = DataLoader(_val_dataset, batch_size=_batch_size, shuffle=False)

    # 损失函数 模型 优化器
    _loss_fn = torch.nn.MSELoss(reduction="mean")
    _model = LinearRegression()
    _optimizer = torch.optim.Adam(_model.parameters(), 
                                  lr=_learning_rate,betas=(0.9, 0.999),  # 一阶与二阶动量的衰减系数
                                  eps=1e-8,  # 防止除零的小常数
                                  weight_decay=1e-2  # L2 正则化)
                                    )

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
    print("end")


