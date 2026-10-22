"""教程第四节：手动实现带截距的一元线性回归梯度下降，动态展示拟合直线。"""
import dataset
import matplotlib.pyplot as plt
import numpy as np

xs, ys = dataset.get_beans2(100)

# 配置图像
plt.title("Size-Toxicity Function", fontsize=14) #设置图像名称
plt.xlabel("Beans Size")  # 设置横坐标的名称
plt.ylabel("Toxicity")  # 设置纵坐标的名称


# 学习率
_learning_rate = 0.01
w = 0.1
# 截距
b = 0.1

for _ in range(100):
    for i in range(100):
        x = xs[i]
        y = ys[i]

        # k 斜率
        k = 2 * x**2 * w + 2* x *b - 2 * x * y
        # k = 2 * (x ** 2) * w + (-2 * x * y)
        # db = 2 * b + 2 * x * 2 - 2 * y
        db = 2 * b + 2 * x * w - 2 * y
        w = w - _learning_rate * k


        b = b - _learning_rate * db

    plt.clf() # 清空窗口
    plt.xlim(0, 1)
    plt.ylim(0, 1.2)

    plt.scatter(xs, ys)
    y_pre = w * xs + b

    plt.plot(xs, y_pre)
    plt.pause(0.01)  # 暂停 0.01s

# plt.scatter(xs, ys)
# plt.show()  # 训练结束后保留窗口

