"""小蓝吃球：手写线性回归练习，不使用机器学习框架。"""
# 手动实现Ronsenblatt 感知器
import dataset
from matplotlib import pyplot as plt

xs, ys = dataset.get_beans(100)
# print(xs)
# print(ys)

# 配置图像
plt.title("Size-Toxicity Function", fontsize=12)  # 设置图像名称
plt.xlabel("Bean Size")  # 设置横坐标的名称
plt.ylabel("Toxicity")  # 设置纵坐标的名称

plt.scatter(xs, ys)


w = 0.5

for _ in range(100):
    for i in range(100):
        x = xs[i]
        y = ys[i]

        y_pre = w * x
        e = y - y_pre

        alpha = 0.05

        w = w + alpha * e * x

y_pre = w * xs

# print(y_pre)

plt.plot(xs, y_pre)

plt.show()