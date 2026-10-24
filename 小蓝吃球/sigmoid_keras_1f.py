"""单输入特征、无隐藏层的 Keras Sigmoid 二分类练习。

使用 Keras 自动求导，由优化器更新权重和偏置。
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


import dataset
import plot_utils

m = 100
X, Y = dataset.get_beans3(m)
# 配置图像
plt.title("Size-Toxicity Function", fontsize=14) #设置图像名称
plt.xlabel("Beans Size")  # 设置横坐标的名称
plt.ylabel("Toxicity")  # 设置纵坐标的名称

plt.scatter(X, Y)

plt.show()

_model = Sequential()

# Dense 是全连接层；一个输入特征连接到一个 Sigmoid 输出神经元
_model.add(Dense(units=1,activation="sigmoid", input_dim=1))

# 使用均方误差（MSE）作为损失函数
_model.compile(
    loss="mean_squared_error",
    optimizer=SGD(learning_rate=0.05),
    metrics=["accuracy"])

_model.fit(X, Y, epochs=5000,   batch_size=10)
_pres = _model.predict(X)

plot_utils.show_scatter_curve(X, Y, _pres)

