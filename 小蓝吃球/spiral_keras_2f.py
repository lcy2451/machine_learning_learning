"""双输入特征的 Keras 螺旋二分类练习。

使用 get_beans6 生成两类带坐标噪声的螺旋（蚊香）数据，两个输入特征为平面坐标。
网络结构为 2 → 8 → 8 → 8 → 1，三个隐藏层使用 ReLU，输出层使用 Sigmoid。
通过 Keras 自动求导，以 MSE 损失和 SGD 优化器训练，并展示样本散点及预测曲面。
当前仅统计训练集准确率，未划分独立验证集或测试集。
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

import dataset
import plot_utils

if __name__ == "__main__":
    m = 100
    _X, _Y = dataset.get_beans6(m)
    plot_utils.show_scatter(_X, _Y)

    _model = Sequential()
    _model.add(Dense(units=8, input_dim=2, activation="relu"))
    _model.add(Dense(units=8, activation="relu"))
    _model.add(Dense(units=8, activation="relu"))
    _model.add(Dense(units=1, activation="sigmoid"))

    # 使用均方误差（MSE）作为损失函数
    _model.compile(
        loss="mean_squared_error",
        optimizer=SGD(learning_rate=0.05),
        metrics=["accuracy"])

    _model.fit(_X, _Y, epochs=5000, batch_size=10)

    # _pres = _model.predict(X)

    plot_utils.show_scatter_surface2(_X, _Y, _model)
