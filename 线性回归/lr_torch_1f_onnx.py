from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader
import onnxruntime as ort


def train(x:torch.Tensor, y:torch.Tensor, loss_fn, plot_df:pd.Series | pd.DataFrame, onnx_path:Path,
          number_epochs = 20, batch_size = 50, learning_rate = 0.001,
          save_onnx=False) -> torch.nn.Linear:
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
            # print('哈哈哈 ', loss_fn)

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

    # 把模型切换到评估模式（Evaluation Mode）。
    model.eval()

    # 示例输入：一条行程，一个特征（里程），形状为 [1, 1]。
    # 用于确定模型的输入格式，不参与训练，也不限制只能预测 1 英里。
    # 当前没有设置动态形状，导出后每次输入一条行程。

    if save_onnx:
        example_input = torch.tensor([[1.0]], dtype=torch.float32)
        # 将计算结构和已训练好的参数保存为 ONNX，默认 export_params=True。
        # 保存的是预测模型，不包含读取 CSV、训练循环或优化器状态。
        torch.onnx.export(

            model,                              # 使用刚训练好的模型
            (example_input, ),                  # 单元素元组，表示模型只有一个输入张量
            onnx_path,                          # 导出文件路径，同名文件会被覆盖
            input_names=["TRIP_MILES"],          # 读取模型预测时使用这个输入名
            output_names=["FARE"],               # 读取模型预测时使用这个输出名
            dynamo=True,                        # 使用基于 torch.export 的 ONNX 导出方式
            external_data=False                 # 参数保存在 .onnx 内，不单独生成 .onnx.data

        )


    # 返回内存中的 PyTorch 模型；仅当 save_onnx=True 时更新 ONNX 文件。
    return model


def predict(onnx_path:Path):

    # 从文件加载模型并创建推理会话，使用 CPU 执行预测，不会重新训练。
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    # ONNX Runtime 接收 NumPy 数组：14.4 英里，形状 [1, 1]。
    # float32 对应导出时示例输入的 torch.float32。
    x = np.array([[14.4]], dtype=np.float32)

    # 第一个参数指定要取出的输出；字典把输入名映射到实际输入数据。
    # 名字必须与导出时的 input_names、output_names 对应。
    outputs = session.run(["FARE"], {"TRIP_MILES": x})

    # outputs 是输出列表；outputs[0] 是 FARE 数组，形状为 [1, 1]。
    # [0, 0] 取第一条行程的第一个输出值，也就是预测车费。
    fare = outputs[0][0, 0]
    print(f"fare {fare:.4f}")

if __name__ == '__main__':
    # CSV 路径相对于运行时工作目录，运行本脚本时需将工作目录设为“线性回归”。
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
    _number_epochs = 60
    _batch_size = 50
    # 学习率
    _learning_rate = 0.001
    # print(type(x))

    # 保留原练习的绘图样本参数；当前 train() 未使用它，也不绘图。
    _plot_df = _training_df.sample(
        n=min(200, len(_training_df)),
        random_state=42
    )

    # 损失函数：MSE
    _loss_fn = torch.nn.MSELoss()
    # MAE：平均绝对误差
    # _loss_fn = torch.nn.L1Loss()
    # 模型路径以当前脚本所在目录为基准，不受运行时工作目录影响。
    _onnx_path = model_path = Path(__file__).resolve().parent / "models" / "lr_torch_1f.onnx"
    # 自动创建父目录；目录已经存在时也不会报错。
    _onnx_path.parent.mkdir(parents=True, exist_ok=True)


    # 直接运行脚本会先训练并导出模型，再读取刚保存的文件进行预测。
    model:torch.nn.Linear = train(
        _x,
        _y,
        _loss_fn,
        _plot_df,
         _onnx_path,
        _number_epochs,
        _batch_size,
        _learning_rate,
        save_onnx=True
       )
    predict(_onnx_path)
