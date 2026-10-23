"""手动实现双输入特征、无隐藏层的 Sigmoid 二分类训练。

使用平方误差损失，通过链式法则计算两个权重和偏置的梯度，并展示训练前后的预测曲面。
"""
import dataset
import numpy as np
import plot_utils

m = 100
# 学习率
_learning_rate = 0.01
xs, ys = dataset.get_beans5(m)

plot_utils.show_scatter(xs, ys)

w1 = 0.1
w2 = 0.1
# 偏置参数
b = 0.1

_x1s = xs[:,0]
_x2s = xs[:,1]

def forward_propagation(x1s, x2s):
    z = w1 * x1s + w2 * x2s + b
    a = 1 / (1 + np.exp(-z))
    return a

plot_utils.show_scatter_surface(xs, ys, forward_propagation)

for _ in range(500):
    for i in range(m):
        x = xs[i]
        y = ys[i]
        x1 = x[0]
        x2 = x[1]

        a = forward_propagation(x1, x2)

        e = (y - a) ** 2

        deda = -2 * (y - a)
        dadz = a * (1 - a)
        dzdw1 = x1
        dzdw2 = x2
        dzdb = 1

        dedw1 = deda * dadz * dzdw1
        dedw2 = deda * dadz * dzdw2
        dedb = deda * dadz * dzdb

        w1 = w1 - _learning_rate * dedw1
        w2 = w2 - _learning_rate * dedw2
        b = b - _learning_rate * dedb

plot_utils.show_scatter_surface(xs, ys, forward_propagation)
