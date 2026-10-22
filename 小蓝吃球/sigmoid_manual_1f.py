"""手动实现单输入特征的 Sigmoid 二分类训练。

使用平方误差损失，通过链式法则计算权重和截距的梯度，并动态展示预测曲线。
"""
import dataset
import matplotlib.pyplot as plt
import numpy as np

xs, ys = dataset.get_beans3(100)

# 配置图像
plt.title("Size-Toxicity Function", fontsize=14) #设置图像名称
plt.xlabel("Beans Size")  # 设置横坐标的名称
plt.ylabel("Toxicity")  # 设置纵坐标的名称

plt.scatter(xs, ys)

# 学习率
_learning_rate = 0.01
w = 0.1
# 截距
b = 0.1

z = w * xs + b
#sigmoid函数
a = 1 / (1 + np.exp(-z))

plt.plot(xs, a)
# plt.show()

for _ in range(5000):
    for i in range(100):
        x = xs[i]
        y = ys[i]

        # 对 w 和 b 求 （偏） 导
        z = w * x + b
        a = 1 / (1 + np.exp(-z))

        deda = -2 * (y - a)
        dadz = a * (1 - a)
        dzdw = x

        dedw = deda * dadz * dzdw

        dzdb = 1

        dedb = deda * dadz * dzdb

        w = w - _learning_rate * dedw
        b = b - _learning_rate * dedb

    if _ % 100 == 0:
        plt.clf() # 清空窗口
        plt.xlim(0, 1)
        plt.ylim(0, 1.2)

        plt.scatter(xs, ys)
        z = w * xs + b
        a = 1 / (1 + np.exp(-z))

        plt.plot(xs, a)
        plt.pause(0.01)  # 暂停 0.01s

# plt.scatter(xs, ys)
# plt.show()  # 训练结束后保留窗口

