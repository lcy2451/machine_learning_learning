from pprint import pprint

import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader


def train(x:torch.Tensor, y:torch.Tensor, loss_fn, model, in_feature_names,
          out_feature_names,
          plot_df:pd.Series | pd.DataFrame,
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

            loss_history.append(epoch_loss.item())
            rmse_history.append(epoch_rmse.item())

        print("epoch: ", epoch+1, "/", number_epochs)
        print("loss: ", f"{epoch_loss.item():.4f}")
        print("rmse: ", f"{epoch_rmse.item():.4f}")
        print('\n')

    # 先选中一个元素，再转换成 Python 数字
    weight_miles = model.weight[0, 0].item()
    weight_seconds = model.weight[0, 1].item()
    bias = model.bias.item()

    print("里程权重:", f"{weight_miles:.4f}")
    print("时长权重:", f"{weight_seconds:.4f}")
    print("偏置:", f"{bias:.4f}")

    epochs = range(1, number_epochs + 1)

    seconds = torch.linspace(
        x[:, 1].min().item(),
        x[:, 1].max().item(),
        steps=20
    )

    # 在里程、时长的范围内，各取 20 个数
    miles = torch.linspace(
        x[:, 0].min().item(),
        x[:, 0].max().item(),
        steps=20
    )

    # # 生成网格：每一个里程都搭配全部 20 个时长
    # # 共得到 20 × 20 = 400 组组合
    grid_miles, grid_seconds = torch.meshgrid(
        miles, seconds, indexing="ij"
    )

    # 将网格整理成模型要求的形状：[400, 2]
    # 每一行依然是：[里程, 时长]
    grid_x = torch.stack(
        [grid_miles.reshape(-1), grid_seconds.reshape(-1)],
        dim=1
    )

    model.eval()

    with torch.no_grad():
        # grid_predictions = model(line_x)
        line_y = model(grid_x)

    # print(len(grid_x))
    # print(len(grid_predictions))
    show_predictions_plt(
        epochs=epochs,
        loss_history=loss_history,
        rmse_history=rmse_history,
        x=x, y=y, loss_fn=loss_fn, model=model,
        in_feature_names=in_feature_names,
        out_feature_names=out_feature_names,
        plot_df=plot_df, line_x=grid_x, line_y=line_y)
    # pprint(line_y.shape)
    # pprint(grid_x.shape)


def show_predictions_plt(
        epochs, loss_history, rmse_history, x:torch.Tensor, y:torch.Tensor, loss_fn, model,
        in_feature_names, out_feature_names, plot_df:pd.Series | pd.DataFrame,
        line_x, line_y):
    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]

    # 正常显示负号
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(18, 5))

    axes = [
        fig.add_subplot(2, 3, 1),  # Loss
        fig.add_subplot(2, 3, 2),  # RMSE

    ]

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

    print('\n\n')
    _index = 4
    for _in_features in range(x.shape[1]):
        subplot = fig.add_subplot(2, 3, _index)
        subplot.set_title(f"预测 {in_feature_names[_in_features]}")
        subplot.set_xlabel(in_feature_names[_in_features])
        subplot.set_ylabel(out_feature_names[0])
        subplot.scatter(
            plot_df[in_feature_names[_in_features]],
            plot_df["FARE"],
            alpha=0.5,
            label=f"实际 {out_feature_names[0]}"
        )

        subplot.plot(
            line_x[:, _in_features].numpy(),
            line_y.numpy().flatten(),
            label=f"预计 {out_feature_names[0]}"
        )
        axes.append(subplot)
        _index += 1
        # print(_in_features)

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
    _training_df['COMPANY'] = _training_df['COMPANY'].str.upper().str.replace(',', '', regex=False).str.strip()
    _company = set()
    for i in _training_df['COMPANY']:
        _company.add(i)
    _company = list(_company)

    _index = 0
    for company in _company:
        company_to_int = 0.0
        for s in company:
            company_to_int += ord(s)
        _training_df.loc[_training_df['COMPANY'] == company, 'COMPANY'] = str(company_to_int)

    _training_df['COMPANY'] = _training_df['COMPANY'].astype(float)

    # 可以选择随机抽样， 用一部分数据来训练， 这里我选择用全部的数据
    # 可选
    # training_df = training_df.sample(
    #     n=min(200, len(training_df)),
    #     random_state=42
    # )

    _in_feature_names = ['TRIP_MILES', 'TRIP_SECONDS']
    _out_feature_names = ['FARE']
    _x: torch.Tensor = torch.tensor(
        _training_df[_in_feature_names].values, dtype
        =torch.float)
    # 真实标签 y：车费
    _y:torch.Tensor = torch.tensor(_training_df[_out_feature_names].values, dtype=torch.float)

    # print(x.shape)
    # print(y.shape)
    _number_epochs = 60
    # _number_epochs = 10
    # _number_epochs = 1
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

    # in_features = 2  输入特征数量
    # out_features = 1 输出特征数量
    # _x.shape[1] 就是x输入的数量了
    _model = torch.nn.Linear(in_features=_x.shape[1], out_features=1)
    train(
        _x,
        _y,
        _loss_fn,
        _model,
        _in_feature_names,
        _out_feature_names,
        _plot_df,
        _number_epochs,
        _batch_size,
        _learning_rate)


