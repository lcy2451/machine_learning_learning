"""教程第三节：手动实现一元线性回归的梯度下降，不使用机器学习框架。"""
import dataset
import matplotlib.pyplot as plt
import numpy as np

xs, ys = dataset.get_beans(100)

# 配置图像
plt.title("Size-Toxicity Function", fontsize=14) #设置图像名称
plt.xlabel("Beans Size")  # 设置横坐标的名称
plt.ylabel("Toxicity")  # 设置纵坐标的名称


# 学习率
_learning_rate = 0.1
w = 0.1

for _ in range(100):
    for i in range(100):
        x = xs[i]
        y = ys[i]

        # k 斜率
        k = 2 * (x**2) * w + (-2 * x * y)
        w = w - _learning_rate * k
        
        plt.clf() # 清空窗口
        plt.xlim(0, 1)
        plt.ylim(0, 1.2)

        plt.scatter(xs, ys)
        y_pre = w * xs

        plt.plot(xs, y_pre)
        plt.pause(0.01)  # 暂停 0.01s

# plt.scatter(xs, ys)
# plt.show()  # 训练结束后保留窗口

