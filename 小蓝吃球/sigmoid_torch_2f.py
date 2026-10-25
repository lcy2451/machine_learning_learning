"""双输入特征、无隐藏层的 PyTorch Sigmoid 二分类练习。

对应 get_beans5 数据，使用两个输入特征，通过一个 Sigmoid 输出神经元进行分类。
使用 MSE 损失和 SGD 训练，每轮统计训练集指标，并在二维输入网格上预测、绘制三维曲面。
"""
import os
# 本机 OpenMP 重复加载问题的临时绕过设置，并未消除底层运行库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from collections.abc import Callable

import matplotlib.pyplot as plt
import torch
# 使用 TorchMetrics 统计二分类准确率
from torchmetrics.classification import BinaryAccuracy
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import dataset
import plot_utils

def show_scatter_surface_with_model(X, Y, model):
	"""接收 NumPy 样本，生成二维网格并使用 PyTorch 模型绘制预测曲面。"""

	x = X[:, 0]
	z = X[:, 1]
	y = Y

	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	# ax = Axes3D(fig)
	ax.scatter(x, z, y)

	x = np.arange(np.min(x), np.max(x), 0.1)
	z = np.arange(np.min(z), np.max(z), 0.1)
	x, z = np.meshgrid(x, z)

	X = np.column_stack((x[0], z[0]))

	for j in range(z.shape[0]):
		if j == 0:
			continue
		X = np.vstack((X, np.column_stack((x[0], z[j]))))

	y = model(torch.tensor(X, dtype=torch.float))

	# 解除梯度关联并转成 NumPy 数组，再恢复为曲面网格形状
	y = np.array([y.detach().cpu().numpy()])
	y = y.reshape(x.shape[0], z.shape[1])
	ax.plot_surface(x, z, y, cmap='rainbow')
	plt.show()

def train(train_load:DataLoader, model:torch.nn.Module,
		  optimizer:torch.optim.Optimizer,
		  loss_fn:Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
		  accuracy: BinaryAccuracy,
		  epochs:int):

	for epoch in range(epochs):
		model.train()

		for batch_x, batch_y in train_load:
			# 1. 前向传播
			prediction = model(batch_x)
			# 2. 计算 Loss
			loss = loss_fn(prediction, batch_y)

			# 3. 清空梯度
			optimizer.zero_grad()

			# 4. 反向传播
			loss.backward()

			# 5. 更新参数
			optimizer.step()

		# 把模型切换到评估模式（Evaluation Mode）。
		model.eval()

		# 每个 Epoch 开始评估时重置指标
		accuracy.reset()

		# 关闭梯度记录
		with torch.no_grad():

			epoch_loss = 0
			for val_x_batch, val_y_batch in train_load:
				# 在训练集上评估，不是独立验证集
				predictions = model(val_x_batch)

				batch_loss = loss_fn(predictions, val_y_batch)
				epoch_loss += batch_loss * len(val_x_batch)

				# 更新指标
				probabilities = predictions.flatten()
				targets = val_y_batch.int().flatten()
				accuracy.update(probabilities, targets)

		epoch_loss = epoch_loss / len(train_load.dataset)

		epoch_accuracy = accuracy.compute()
		print(f"epoch: {epoch + 1}/{epochs}"
			  f" train loss: {epoch_loss:.4f}"
			  f" accuracy: {epoch_accuracy:.4f}")

if __name__ == "__main__":
	# 随机种子， 42 是“生命、宇宙以及一切的终极答案”。
	torch.manual_seed(42)

	# 样本数
	m = 100
	# 学习率
	_learning_rate = 0.03
	_batch_size = 10
	_epochs = 50000

	_X, _Y = dataset.get_beans5(m)
	X = torch.tensor(_X, dtype=torch.float) # 每行有两个输入特征，保留 (m, 2) 形状
	Y = torch.tensor(_Y, dtype=torch.float).reshape(-1, 1)

	plot_utils.show_scatter(_X, _Y)

	_model = torch.nn.Sequential(
		torch.nn.Linear(2, 1),
		torch.nn.Sigmoid(),
	)

	# 创建优化器
	_optimizer = torch.optim.SGD(_model.parameters(), lr=_learning_rate)

	# 定义指标
	_acc = BinaryAccuracy()

	# 训练数据集
	_dataset = TensorDataset(X, Y)
	_dataloader = DataLoader(_dataset, batch_size=_batch_size, shuffle=True)

	# loss 使用均方误差
	torch.nn.MSELoss()

	train(train_load=_dataloader, model=_model,
		  optimizer=_optimizer,
		  loss_fn=torch.nn.MSELoss(),
		  accuracy=_acc,
		  epochs=_epochs)

	_model.eval()
	with torch.no_grad():
		_pres = _model(X).cpu().numpy()

	show_scatter_surface_with_model(_X, _Y, _model)
