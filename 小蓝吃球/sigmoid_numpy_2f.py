"""双输入特征的 NumPy Sigmoid 二分类练习。

使用 NumPy 数组运算实现前向传播、手动计算梯度和参数更新。
通过向量点积合并两个特征的加权求和，仍按固定顺序逐样本训练。
"""
from re import X

import dataset
import numpy as np
import plot_utils

m = 100
# 学习率
_learning_rate = 0.01
_X, _Y = dataset.get_beans5(m)

plot_utils.show_scatter(_X, _Y)

W = np.array([0.1, 0.1])
# 偏置参数
B = 0.1


def forward_propagation(x:np.ndarray):
    # z = w1 * x1s + w2 * x2s + b
    # W 是一维数组，W.T 不改变形状；点积完成两个特征的加权求和
    z = x.dot(W.T) + B
    a = 1 / (1 + np.exp(-z))
    return a

plot_utils.show_scatter_surface2(_X, _Y, forward_propagation)

for _ in range(500):
    for i in range(m):
        Xi = _X[i]
        Yi = _Y[i]

        A = forward_propagation(Xi)

        E = (Yi - A) ** 2

        dEdA = -2 * (Yi - A)
        dAdZ = A * (1 - A)
        dZdW = Xi
        dZdB = 1

        dEdw = dEdA * dAdZ * dZdW
        dEdB = dEdA * dAdZ * dZdB

        W = W - _learning_rate * dEdw
        B = B - _learning_rate * dEdB

plot_utils.show_scatter_surface2(_X, _Y, forward_propagation)
