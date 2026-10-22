"""教程第四节：手动计算一元线性回归的均方误差，并展示三维误差曲面。

模型为 y = wx + b，拟合直线不再被限制为经过原点。
三维图的三个轴分别表示权重 w、截距 b 和代价函数的值。
通过扫描不同的 w 和 b 绘制误差曲面，不执行梯度下降训练。
"""
import numpy as np
import dataset
from matplotlib import pyplot as plt


m = 100

xs, ys = dataset.get_beans2(m)

fig = plt.figure()
# ax = Axes3D(fig)
ax = fig.add_subplot(111, projection="3d")
ax.set_zlim(0, 2)

ws = np.arange(-1, 2, 0.1)
bs = np.arange(-2, 2, 0.01)

for b in bs:
    es = []
    for w in ws:
        y_pre = w * xs + b
        e = np.sum((ys - y_pre) ** 2) * (1 / m)
        es.append(e)

    # plt.plot(ws, es)
    # ax.plot(ws, es, b, zdir="y", color="blue")
    ax.plot(ws, es, b, zdir="y")

plt.show()
