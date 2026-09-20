import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader


def train(x:torch.Tensor, y:torch.Tensor, loss_fn, plot_df:pd.Series | pd.DataFrame,
          number_epochs = 20, batch_size = 50, learning_rate = 0.001,
          ):
    # 随机种子， 42 是“生命、宇宙以及一切的终极答案”。
    torch.manual_seed(42)

    # 把特征和标签配对 不会在这里训练模型，也不会随机打乱数据。它只是将多个 Tensor 组织为一个可以按索引访问的数据集。
    dataset = TensorDataset(x, y)

    # 创建数据加载器
    # 假设你有 200 条训练数据，而模型每次只需要处理 50 条。
    # DataLoader 就可以把数据分成 4 个 Batch（批次），让训练循环依次获取。
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    print("总样本数:", len(dataset))
    print("每个 Epoch（周期） 的 Batch 数:", len(train_loader))

    # 创建线性回归模型
    # in_features = 1  输入特征数量
    # out_features = 1 输出特征数量
    model = torch.nn.Linear(in_features=1, out_features=1)

    # 优化器：RMSprop
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    print("weight  ", model.weight)
    print("bias  ",model.bias)

    # 当前选择的训练损失
    loss_history = []

    # 计算 RMSE 用于观察预测误差
    rmse_history = []

    # 开始按 Epoch 训练
    for epoch in range(number_epochs):
        # 模型切换到训练模式。
        # 注意：model.train() 本身不会执行训练，也不会自动计算梯度或更新参数！
        model.train()

        # 从 train_loader 中逐批取出训练数据，每次得到一组输入特征 batch_x 和对应的真实标签 batch_y。
        for batch_x, batch_y in train_loader:
            # 在出租车项目中：
            #     batch_x：50条行程里程（TRIP_MILES），模型的输入。
            #     batch_y：50条真实车费（FARE），模型的标签。

            # 使用 50 条行程里程预测车费
            predictions = model(batch_x)

            # 将预测车费与 50 条真实车费比较, 得到了当前损失值
            loss = loss_fn(predictions, batch_y)
            # 清空上一次计算得到的梯度（Gradient），为当前Batch的反向传播做准备。
            optimizer.zero_grad()

            # 执行反向传播（Backpropagation），计算Loss对模型参数的梯度（Gradient）。
            loss.backward()

            # 根据刚刚计算出的梯度，真正更新模型的Weight（权重）和Bias（偏差）。
            optimizer.step()

        # 把模型切换到评估模式（Evaluation Mode）。
        model.eval()

        # 关闭梯度记录
        with torch.no_grad():

            # 重新预测全部数据, 和前面predictions = model(batch_x)的区别是这里的是全部的数据， 之前是单个批次的数据
            predictions = model(x)

            # 计算整个训练集的损失
            epoch_loss = loss_fn(predictions, y)
            # 计算RMSE 用于观察预测误差的
            # epoch_rmse = torch.sqrt(epoch_loss)
            # 评估指标：RMSE
            if isinstance(loss_fn, torch.nn.MSELoss):
                epoch_rmse = torch.sqrt(epoch_loss)
            elif isinstance(loss_fn, torch.nn.L1Loss):
                epoch_rmse = torch.sqrt(torch.mean((predictions - y) ** 2))
            print('哈哈哈 ', loss_fn)

            loss_history.append(epoch_loss.item())
            rmse_history.append(epoch_rmse.item())

        print("epoch: ", epoch+1, "/", number_epochs)
        print("loss: ", f"{epoch_loss.item():.4f}")
        print("rmse: ", f"{epoch_rmse.item():.4f}")
        print('\n')

    weight = model.weight.item()
    bais = model.bias.item()

    print("weight: ", f"{weight:.4f}")
    print("bias: ", f"{bais:.4f}")

    print("weight: ", f"{weight:.4f}")
    print("bias: ", f"{bais:.4f}")

    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]

    # 正常显示负号
    plt.rcParams["axes.unicode_minus"] = False

    # 创建一个窗口，包含 3 张子图
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 5)
    )

    epochs = range(1, number_epochs + 1)

    if isinstance(loss_fn, torch.nn.L1Loss):
        loss_name = "MAE"
    elif isinstance(loss_fn, torch.nn.MSELoss):
        loss_name = "MSE"
    else:
        loss_name = type(loss_fn).__name__

    # 第一张：Loss
    axes[0].plot(epochs, loss_history, marker="o")
    axes[0].set_title("Training Loss 训练损失")
    axes[0].set_xlabel("Epoch 周期")
    axes[0].set_ylabel(loss_name)
    axes[0].grid(True)

    # 第二张：RMSE
    axes[1].plot(epochs, rmse_history, marker="o")
    axes[1].set_title("Training RMSE 训练集均方根误差")
    axes[1].set_xlabel("Epoch 周期")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(True)

    # plot_df = training_df

    # 生成 100 个均匀分布的行程里程，用于绘制模型的预测直线。
    line_x = torch.linspace(
        x.min().item(),
        x.max().item(),
        steps=100
    ).reshape(-1, 1)

    # 把模型切换到评估模式（Evaluation Mode）。
    model.eval()

    # 预测车费
    with torch.no_grad():
        line_y = model(line_x)
        predict1 = model(torch.tensor([1], dtype=torch.float))

    axes[2].scatter(
        plot_df["TRIP_MILES"],
        plot_df["FARE"],
        alpha=0.5,
        label="Actual Fare 实际票价"
    )

    # 用红色五角星标记预测点
    axes[2].scatter(
        1,
        predict1,
        color="red",
        marker="*",
        s=200,
        zorder=5,
        label="预测英里"
    )

    axes[2].plot(
        line_x.numpy().flatten(),
        line_y.numpy().flatten(),
        label="Predicted Fare 预计票价"
    )

    axes[2].set_title("Linear Regression 线性回归")
    axes[2].set_xlabel("Trip Miles 行程里程")
    axes[2].set_ylabel("Fare 票价")
    axes[2].legend()
    axes[2].grid(True)

    # 自动调整子图间距
    plt.tight_layout()

    # 只显示一次
    plt.show()


if __name__ == '__main__':
    # @title
    chicago_taxi_dataset = pd.read_csv("../data/chicago_taxi_train.csv")
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

    # print(x.shape)
    # print(y.shape)
    _number_epochs = 20
    _batch_size = 50
    # 学习率
    _learning_rate = 0.001
    # print(type(x))

    # 预测直线
    _plot_df = _training_df.sample(
        n=min(200, len(_training_df)),
        random_state=42
    )

    # 损失函数：MSE
    # loss_fn = torch.nn.MSELoss()
    # MAE：平均绝对误差
    _loss_fn = torch.nn.L1Loss()

    train(
        _x,
        _y,
        _loss_fn,
        _plot_df,
        _number_epochs,
        _batch_size,
        _learning_rate)


