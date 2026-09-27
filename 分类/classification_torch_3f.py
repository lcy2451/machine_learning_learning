from pathlib import Path

import onnx
import pandas as pd
import torch.nn
import onnxruntime as ort



def train(in_features:torch.Tensor, in_labels:torch.Tensor, loss_fn, model,
          number_epochs = 20, batch_size = 50, learning_rate = 0.001,
          classification_threshold=0.35):


    # 创建优化器 Adam
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


    # 创建数据集
    dataset = torch.utils.data.TensorDataset(
        in_features,
        in_labels)

    # 数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # 开始按 Epoch 训练
    for epoch in range(number_epochs):
        # 模型切换到训练模式。
        # 注意：model.train() 本身不会执行训练，也不会自动计算梯度或更新参数！
        model.train()

        epoch_loss = 0.0

        for batch_x, batch_y in dataloader:
            # 1. 清空上一轮梯度
            optimizer.zero_grad()

            # 2. 前向传播
            logits = model(batch_x)

            # 3. 计算损失
            loss = loss_fn(logits, batch_y)

            # 4. 反向传播，计算梯度
            loss.backward()

            # 5. 根据梯度更新模型参数
            optimizer.step()

            epoch_loss += loss.item()

        # 切换模型到评估模式
        model.eval()

        average_loss = epoch_loss / len(dataloader)

        with torch.no_grad():
            logits = model(in_features)

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= classification_threshold
            ).float()

            accuracy = (predictions == in_labels).float().mean()

        print(
            "\n"
            f"Epoch {epoch + 1} / {number_epochs}  "
            f"Loss {average_loss:.4f}  "
            f"Accuracy {accuracy.item():.4f}"
        )

def save_onnx(model, filename:Path):
    dummy_input = torch.randn(1, 3)

    torch.onnx.export(
        model,
        (dummy_input, ),
        filename,
        input_names=["features"],
        output_names=["logits"],
        external_data=False,
        dynamo = True

    )

    __model = onnx.load(filename)
    onnx.checker.check_model(__model)


def predict(onnx_path:Path, in_features: torch.Tensor, validation_label: torch.Tensor):
    # 从文件加载模型并创建推理会话，使用 CPU 执行预测，不会重新训练。
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    # ONNX Runtime 接收 NumPy 数组
    # 当前模型一次预测一条，形状为 (1, 3)
    x = in_features.detach().cpu().numpy()

    logits = session.run(["logits"], {"features": x})[0]

    # 线性层输出的 logits 还不是概率，需要经过 sigmoid
    probability = torch.sigmoid(torch.from_numpy(logits)).item()

    # 与训练时使用相同的分类阈值
    prediction = 1 if probability >= 0.35 else 0
    rice_class = "Cammeo" if prediction == 1 else "Osmancik"
    label_class = "Cammeo" if int(validation_label.item()) == 1 else "Osmancik"

    print('\n')
    print('=' * 10)
    print(f"输入 {in_features}", end='')
    print(f"   属于 Cammeo 的概率：{probability:.4f}")
    print(f"预测类别：{rice_class}")
    print(f"实际的类别 {label_class}")
    print('=' * 10)
    print('\n')

    return probability, rice_class


if __name__ == '__main__':
    chicago_taxi_dataset = pd.read_csv("../data/Rice_Cammeo_Osmancik.csv")

    # Area 面积
    # Eccentricity 离心率
    # Major_Axis_Length 长度长度
    _training_df:pd.Series | pd.DataFrame = chicago_taxi_dataset.loc[
        :,( "Area", "Eccentricity", "Major_Axis_Length", "Class")
    ]

    _training_df["Class_Bool"] = (
            _training_df["Class"] == "Cammeo"
    ).astype(int)


    # 标签
    _train_labels:torch.Tensor = torch.Tensor(_training_df[["Class_Bool"]].values)

    # 输入特征
    features = _training_df[["Area", "Eccentricity", "Major_Axis_Length"]]
    # Z-score 标准化，让三个特征处于相近的数值尺度
    features = (features - features.mean()) / features.std()
    _in_features:torch.Tensor = torch.Tensor(features.values)

    _number_epochs = 60
    _batch_size = 50
    # 学习率
    _learning_rate = 0.001

    # 分类阈值
    _classification_threshold = 0.35

    # 随机种子， 42 是“生命、宇宙以及一切的终极答案”。
    torch.manual_seed(42)
    # 创建模型
    _model = torch.nn.Linear(in_features=3, out_features=1)

    # 损失函数 AI比较推荐BCEWithLogitsLoss
    _loos_fn = torch.nn.BCEWithLogitsLoss()

    # train(_in_features,
    #       _train_labels,
    #       _loos_fn,
    #       _model,
    #       number_epochs=_number_epochs,
    #       batch_size=_batch_size,
    #       learning_rate=_learning_rate,
    #       classification_threshold=_classification_threshold)
    #
    _onnx_path = model_path = Path(__file__).resolve().parent / "models" / "classification_torch_3f.onnx"
    # 自动创建父目录；目录已经存在时也不会报错。
    _onnx_path.parent.mkdir(parents=True, exist_ok=True)
    # save_onnx(_model, _onnx_path)
    #

    for _in_feature, _validation_label in zip(_in_features, _train_labels):
        input_data = _in_feature.unsqueeze(0)
        predict(_onnx_path, input_data, _validation_label)
