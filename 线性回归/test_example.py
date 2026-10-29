import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# 随机种子， 42 是“生命、宇宙以及一切的终极答案”。
torch.manual_seed(42)

# 超参数
# 训练轮数
epochs = 20
# 批量大小
batch_size = 50
# 学习率
learning_rate = 0.001

# 读取数据集
chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

# TRIP_MILES  行程里程
# TRIP_SECONDS 行程时长（秒）
# FARE 票价
# COMPANY  公司
# PAYMENT_TYPE  支付方式
# TIP_RATE 小费
training_df:pd.Series | pd.DataFrame = chicago_taxi_dataset.loc[
    :, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')]

# 把 Pandas 数据转换成 PyTorch 的 Tensor（张量）。
# 输入特征 X：行程里程
x:torch.Tensor = torch.tensor(training_df[['TRIP_MILES']].values, dtype=torch.float)
# 真实标签 y：车费
y:torch.Tensor = torch.tensor(training_df[['FARE']].values, dtype=torch.float)

# 训练数据集
dataset = TensorDataset(x, y)
# 创建数据加载器
# 假设你有 200 条训练数据，而模型每次只需要处理 50 条。
# DataLoader 就可以把数据分成 4 个 Batch（批次），让训练循环依次获取。
train_load = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
    )

# 损失函数
# L1 loss：绝对误差之和
loss_fn = torch.nn.L1Loss(reduction="sum")

# 创建模型
model = torch.nn.Linear(in_features=1, out_features=1)

# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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

    # 关闭梯度记录
    with torch.no_grad():

        epoch_loss = 0
        
        for eval_x, eval_y in train_load:
            # 本例暂不划分数据集，直接计算模型在训练集上的平均损失
            predictions = model(eval_x)

            batch_loss = loss_fn(predictions, eval_y)
            epoch_loss += batch_loss.item()

        # 循环结束后，计算每个样本的平均损失
        epoch_loss = epoch_loss / len(train_load.dataset)
        print(f"epoch: {epoch+1}/{epochs}, loss: {epoch_loss:.4f}")

w = model.weight.item()
b = model.bias.item()

#  下面是线性回归计算， y=b+wx
y_pre = b + w * x

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


plt.plot(x, y_pre, color="red")
plt.scatter(x, y, s=10, alpha=0.3)
plt.show()
