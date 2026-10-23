"""手动实现单输入特征的 Sigmoid 二分类训练。

使用平方误差损失，通过链式法则计算权重和截距的梯度，并动态展示预测曲线。
"""
import dataset
import matplotlib.pyplot as plt
import numpy as np

xs, ys = dataset.get_beans4(100)

# 配置图像
plt.title("Size-Toxicity Function", fontsize=14) #设置图像名称
plt.xlabel("Beans Size")  # 设置横坐标的名称
plt.ylabel("Toxicity")  # 设置纵坐标的名称

plt.scatter(xs, ys)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 第一层
# 第一个神经元
w11_1 = np.random.rand()
b1_1 = np.random.rand()
# 第二个神经元
w12_1 = np.random.rand()
b2_1 = np.random.rand()

# 第二层
w11_2 = np.random.rand()
w21_2 = np.random.rand()
b1_2 = np.random.rand()

# 前向传播
def forward(xs):
    z1_1 = w11_1 * xs + b1_1
    a1_1 = sigmoid(z1_1)

    z2_1 = w11_2 * xs + b2_1
    a2_1 = sigmoid(z2_1)

    z1_2 = w11_2 * a1_1 + w21_2 * a2_1 + b1_2
    a1_2 = sigmoid(z1_2)
    return a1_2, z1_2, a2_1, z2_1, a1_1, z1_1

a1_2, z1_2, a2_1, z2_1, a1_1, z1_1 = forward(xs)
# 学习率
_learning_rate = 0.01
# w = 0.1
# # 截距
# b = 0.1

# z = w * xs + b
#sigmoid函数
# a = 1 / (1 + np.exp(-z))

# plt.plot(xs, a1_2)
# plt.show()

for _ in range(5000):
    for i in range(100):
        x = xs[i]
        y = ys[i]

        # 先来一次前向传播
        a1_2, z1_2, a2_1, z2_1, a1_1, z1_1 = forward(x)
        # 反向传播
        # 误差代价e
        e = (y - a1_2)  ** 2

        deda1_2 = -2 * (y - a1_2)

        da1_2dz1_2 = a1_2 * (1 - a1_2)

        dz1_2dw11_2 = a1_1
        dz1_2dw21_2 = a2_1

        dedw11_2 = deda1_2 * da1_2dz1_2 * dz1_2dw11_2
        dedw21_2 = deda1_2 * da1_2dz1_2 * dz1_2dw21_2

        dz1_2db1_2 = 1
        dedb1_2 = deda1_2 * da1_2dz1_2 * dz1_2db1_2

        dz1_2da1_1 = w11_2
        da1_1dz1_1 = a1_1
        dz1_1dw11_1 = x
        dedw11_1 = deda1_2 * da1_2dz1_2 * dz1_2da1_1 * da1_1dz1_1 * dz1_1dw11_1
        dz1_1db1_1 = 1
        dedb1_1 = deda1_2 * da1_2dz1_2 * dz1_2da1_1 * da1_1dz1_1 * dz1_1db1_1

        dz1_2da2_1 = w11_2
        da2_1dz2_1 = a2_1 * (1 - a2_1)
        dz2_1dw11_1 = x
        dedw12_1 = deda1_2 * da1_2dz1_2 * dz1_2da1_1 * da2_1dz2_1 * dz2_1dw11_1
        dz2_1db2_1 = 1
        dedb2_1 = deda1_2 * da1_2dz1_2 * dz1_2da1_1 * da2_1dz2_1 * dz2_1db2_1

        w11_2 = w11_2 - _learning_rate * dedw11_2
        w21_2 = w21_2 - _learning_rate * dedw21_2
        b1_2 = b1_2 - _learning_rate * dedb1_2

        w12_1 = w12_1 - _learning_rate * dedw12_1
        b2_1 = b2_1 - _learning_rate * dedb2_1

        w11_1 = w11_1 - _learning_rate * dedw11_1
        b1_1 = b1_1 - _learning_rate * dedb1_1

    if _ % 100 == 0:
        plt.clf() # 清空窗口
        # plt.xlim(0, 1)
        # plt.ylim(0, 1.2)

        plt.scatter(xs, ys)
        a1_2, z1_2, a2_1, z2_1, a1_1, z1_1 = forward(xs)

        plt.plot(xs, a1_2)
        plt.pause(0.01)  # 暂停 0.01s

# plt.scatter(xs, ys)
# plt.show()  # 训练结束后保留窗口

