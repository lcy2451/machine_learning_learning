# 损失函数loss

[线性回归：损失](https://developers.google.com/machine-learning/crash-course/linear-regression/loss)
[常见的损失函数(loss function)总结](https://zhuanlan.zhihu.com/p/58883095)
[《动手学深度学习》](https://zh.d2l.ai/chapter_preface/index.html)
[loss functions Pytorch)](https://docs.pytorch.org/docs/2.15/nn.html#loss-functions)
[Pytorch Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

在线性回归中，常见的损失函数有  [L1Loss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.L1Loss.html#torch.nn.L1Loss)、L2、MAE、MSE 和 RMSE。


==当一次处理多个示例时，我们建议对所有示例的损失进行平均，无论使用 MAE、MSE 还是 RMSE。==

安装油耗数据集
[Auto MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg)
```python
pip install ucimlrepo
```

# Choosing a loss  选择损失
在数据中， 汽车的重量通常在 2000 到 5000 磅之间，油耗通常在每加仑 8 到 50 英里之间。一辆 8000 磅重的汽车，或者一辆油耗为每加仑 100 英里的汽车，都超出了典型范围，会被视为==异常值== 。

均方误差 (MSE) 会使模型更倾向于异常值， MSE 损失使模型更接近异常值。
平均绝对误差 (MAE) 则不会， MAE 损失使模型远离异常值。
L 2 损失对异常值的惩罚远高于 L 1 损失。


### L1 Loss 函数
[PyTorch torch.nn.L1Loss 函数 菜鸟](https://www.runoob.com/pytorch/pytorch-torch-nn-l1loss.html)
[L1Loss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.L1Loss.html#torch.nn.L1Loss)
##### 参数:
- reduction （ str ， 可选 ）， 'mean' ：输出结果的总和将除以输出结果中的元素个数；'sum' ：输出结果的总和， 不求平均；

torch.nn.L1Loss 是 PyTorch 中的 L1 损失函数，也称为平均绝对误差（MAE）。

