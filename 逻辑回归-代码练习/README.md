# 逻辑回归-代码练习
Sigmoid 函数可以将线性模型的输出映射到 0～1 之间，使模型能够输出概率。因此，在原有线性模型的基础上加入 Sigmoid 函数，就可以构建一个逻辑回归模型。

本次练习使用 [钞票真伪鉴定](https://archive.ics.uci.edu/dataset/267/banknote)

完整的代码在[logistic_train.py](logistic_train.py)

逻辑回归和线性回归的区别是逻辑回归多了一个Sigmoid函数， 在模型结构上，可以使用 Sequential 将线性层和 Sigmoid 组合起来，构建一个简单的逻辑回归模型。

也有另外的方式， 比如我使用 BCEWithLogitsLoss 这个损失函数， 因为 BCEWithLogitsLoss 内部结合了 Sigmoid 和 Binary Cross Entropy（交叉熵损失），所以不需要在模型中额外添加 Sigmoid，就可以直接使用线性层的输出计算损失, 这样只要在线性回归代码的基础上， 修改损失函数就可以把一个线性回归模型改成逻辑回归模型
```python
loss_fn = torch.nn.BCEWithLogitsLoss()
```
接下来检查模型能否输出 0～1 之间的概率。由于模型本身没有添加 Sigmoid，因此预测时需要手动进行转换。
```python
_model.eval()
with torch.no_grad():
    for X, y in test_loader:
        logits = _model(X[0])

        # 使用 sigmoid 将 Logits 转换为 0~1 之间的概率
        prob = torch.sigmoid(logits)

        print(f"prob {prob[0]:.4f}  y {y[0][0]}")

```

输出是
````python
prob 0.9999  y 1.0
prob 1.0000  y 1.0
prob 0.9978  y 1.0
prob 0.9816  y 1.0
prob 0.0000  y 0.0
````
可以看到，模型预测的概率与真实标签比较接近，说明模型能够区分钞票的真假。
