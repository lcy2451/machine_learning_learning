"""双输入特征、无隐藏层的 Keras Sigmoid 二分类练习。

使用 Keras 自动求导，由优化器更新权重和偏置。
使用 get_beans5 数据，通过两个输入特征预测分类概率，并展示预测曲面。
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


import dataset
import plot_utils

m = 100
X, Y = dataset.get_beans5(m)
plot_utils.show_scatter(X, Y)

plt.show()

_model = Sequential()

# Dense 是全连接层；两个输入特征连接到一个 Sigmoid 输出神经元
_model.add(Dense(units=1,activation="sigmoid", input_dim=2))
# _model.add(Dense(units=1,activation="sigmoid", input_dim=1))


# 使用均方误差（MSE）作为损失函数
_model.compile(
    loss="mean_squared_error",
    optimizer=SGD(learning_rate=0.05),
    metrics=["accuracy"])

_model.fit(X, Y, epochs=5000,  batch_size=10)
# _pres = _model.predict(X)

plot_utils.show_scatter_surface2(X, Y, _model)

