"""教程第二节：损失函数  均方误差（MSE）手写练习。"""
import dataset
import matplotlib.pyplot as plt
import numpy as np

xs, ys = dataset.get_beans(100)

# 配置图像
plt.title("Size-Toxicity Function", fontsize=14) #设置图像名称
plt.xlabel("Beans Size")  # 设置横坐标的名称
plt.ylabel("Toxicity")  # 设置纵坐标的名称

plt.scatter(xs, ys)

w = 0.1
y_pre = w * xs

plt.plot(xs, y_pre)

plt.show()

# es = (ys - y_pre) ** 2
#
# sum_e = (1/100) * np.sum(es)
#
ws = np.arange(0, 3, 0.1)

es = []
for w in ws:
    y_pre = w * xs

    # 代价函数
    e = (1/100) * np.sum((ys - y_pre) ** 2)
    es.append(e)

# 配置图像
# 绘制代价函数
plt.title("cost function", fontsize=14) #设置图像名称
plt.xlabel("w")  # 设置横坐标的名称
plt.ylabel("e")  # 设置纵坐标的名称

plt.plot(ws, es)
plt.show()

w_min = np.sum(xs * ys) / np.sum(xs * xs)
print(f"e 最小点w: {w_min}")

y_pre = w_min * xs

# 配置图像
plt.title("Size-Toxicity Function", fontsize=14) #设置图像名称
plt.xlabel("Beans Size")  # 设置横坐标的名称
plt.ylabel("Toxicity")  # 设置纵坐标的名称

plt.scatter(xs, ys)

plt.plot(xs, y_pre)

plt.show()
