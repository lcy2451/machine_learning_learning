为了解决线形不可分的问题， 以及为了让模型可以解决更复杂的问题， 于是我们开始学习使用神经网络，之前看到的逻辑回归模型就是简单的神经网络。 

## 节点与隐藏层
在神经网络术语中，输入层和输出层之间的附加层被称为隐藏层，这些层中的节点被称为神经元。添加隐藏层和神经元可以让模型变得非常复杂， 这样它就可以拟合更复杂的数据， 不断的增加隐藏层可以做到大力出奇迹， 但隐藏层和神经元不是越多越好， 这会带来过拟合的风险

> 要注意一点， 单纯的加入隐藏层和神经元不会让模型变得非线性， 需要加入非线形的激活函数， 就像之前在逻辑回归中加的 sigmoid函数一样

## 常见激活函数
常用作激活函数的三种数学函数是 sigmoid、tanh 和 ReLU。

sigmoid 之前讨论过， 他是一个s型曲线， 常用于在二分类时把输出映射在0-1。

tanh（双曲正切函数）和sigmoid差不多， 他是把范围映射在-1到1

ReLU （是线性整流函数）和上面两个函数不太一样， 首先他看起来就不平滑，但它在神经网络训练过程中不太容易受到梯度消失问题的影响， 其次他的计算也比较简单。 相比于sigmoid函数，其收敛速度快， 能够最大化发挥神经元的筛选能力，
![ReLU](ReLU.png)

# 多类别分类
以下例子指的是多类别分类模型：
- 这只狗是比格犬、巴吉度猎犬还是寻血猎犬？
- 这朵花是西伯利亚鸢尾、荷兰鸢尾、蓝旗鸢尾还是矮须鸢尾？
- 那架飞机是波音 747、空客 320、波音 777 还是巴西航空工业 190？
- 这是一张苹果、熊、糖果、狗还是鸡蛋的图片？

google机器学习介绍了多分类类别的两种变体：
- 一对多（one-vs.-all ）
- 一对一 （one-vs.-one）（softmax）

# One versus all  一对多
训练四个不同的识别器，每个识别器回答一个不同的是/否问题

下图解释了分类器都干了什么
![一对多](一对多.png)
当类别数量增多时， 这种识别器的效率会越来越低

我们可以使用深度神经网络创建一个效率显著更高的一对多模型，其中每个输出节点代表一个不同的类别。下图说明了这种方法。
![神经网络模型完成相同的一对多分类任务](神经网络模型完成相同的一对多分类任务.png)
该模型预测图像为梨的概率为 84%，为葡萄的概率为 7%。

# One versus one (softmax)  一对一（softmax）

上面使用深度神经网络创建的一对多模型使用的是 sigmoid激活函数， 他输出概率的总和不是1， 而是1.43， 在一对多方法中，每组二元结果的概率是独立于所有其他组确定的。也就是说，我们在确定“苹果”与“非苹果”的概率时，并未考虑其他水果选项（“橙子”、“梨”或“葡萄”）的可能性。
如果希望所有类别的输出概率总和为 1，可以考虑使用 Softmax 激活函数。Softmax 会综合考虑所有类别的输出，将它们转换为概率分布。

> 要注意的是， Google说过One versus one（一对一），通常称为 softmax实际并不严谨， Google也没有讲传统的One versus one怎么搞， 只是讲了Softmax， 我这里也不对One versus one（一对一）做学习

---
# 代码部分
既然Google使用的是图像， 我也用图像训练
我使用Fashion-MNIST 服饰图像分类数据集， 他可以用PyTorch 直接加载
完整代码[fashion_mnist_torch.py](fashion_mnist_torch.py)

```python
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def target_transform(y: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(
        torch.tensor(y, dtype=torch.int64),
        num_classes=10
    ).float()

if __name__ == "__main__":


    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
        target_transform=target_transform
    )

    test_dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
        target_transform=target_transform
    )

    plt.imshow(train_dataset[1][0].squeeze(), cmap=plt.cm.binary)
    plt.show()
```
![MNIST数据](MNIST数据.png)
transforms.ToTensor() 是 torchvision 提供的一个图片转换工具， 将 PIL 图像或 ndarray 转换为张量，并相应地缩放数值， 注意他会缩放数值， 这样他可以不用归一化了， 但不是所有情况下都会返回归一化数值。参考文档：
>https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html
```python
transform = transforms.ToTensor()
```

FashionMNIST事用来加载和访问 Fashion-MNIST 数据集， 可以选择训练集和测试集， 通过transform参数传入的工具把图像像素归一化到0-1

```python
def target_transform(y: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(
        torch.tensor(y, dtype=torch.int64),
        num_classes=10
    ).float()
```
这个target_transform是把标签处理为独热编码

----
现在的数据是训练集有60000个数据， 图片大小是28x28。需要在神经网络中展平为784维用来训练。有 10 个类别的标签，所以要输出10个， 之前把标签处理为独热编码是为了把标签处理成10维

下面是创建简单神经网络的代码
```python
class FashionClassifier(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.network = torch.nn.Sequential(
            torch.nn.Flatten(),

            torch.nn.Linear(in_features=784, out_features=128),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=128, out_features=128),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=128, out_features=128),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=128, out_features=10),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # 前向传播
        return self.network(x)
```
我现在创建了三个隐藏层， 每层有128个神经元， 并且每层用ReLU激活函数，然后最后输出10层， 因为数据集有10个类别， 最后加入Sigmoid激活函数， 这符合一对多（one-vs.-all ）的做法

损失函数, 用BCELoss这个损失函数不带Sigmoid， 所以不会和模型的Sigmoid冲突
```python
# 损失函数
loss_fn = torch.nn.BCELoss()    
```

优化器使用AdamW， 因为这次学习没有学优化器
```python
# 优化器
_optimizer = torch.optim.AdamW(
    _model.parameters(),  # 需要优化的模型参数
    lr=_learning_rate,  # 学习率
    betas=(0.9, 0.999),  # 一阶与二阶动量的衰减系数
    eps=1e-8,  # 防止除零的小常数
            weight_decay=1e-2  # 权重衰减（AdamW为解耦方式）
)
```

训练部分和其他一样， 训练部分和其他一样， 神经网络也看loss损失， 而且这是多分类模型， 还可以看准确率、roc、 auc等， 对吧， 但是不止看这个， 但是这个还是太单调  

按照 MNIST数据提供的标签来 检查预测是否准确
| Label | Description（类别名称）  |
| ----- | ------------------ |
| 0     | T-shirt/top（T恤/上衣） |
| 1     | Trouser（裤子）        |
| 2     | Pullover（套头衫）      |
| 3     | Dress（连衣裙）         |
| 4     | Coat（外套）           |
| 5     | Sandal（凉鞋）         |
| 6     | Shirt（衬衫）          |
| 7     | Sneaker（运动鞋）       |
| 8     | Bag（包）             |
| 9     | Ankle boot（短靴）     |


我用下面的代码来打印模型预测的结果， 并且分别尝试 `Sigmoid` 和 `Softmax` 
```python
_model.eval()
with torch.no_grad():
    for X, y in test_loader:
        pred = _model(X)

        for probabilities in pred:
            for name, p in zip(class_names, probabilities):
                print(f"{name:<15}: {p.item():.4f}")

            print(f"Sum: {probabilities.sum().item():.4f}")
            print("-" * 30)

        break
```

用`Sigmoid` 输出的结果是这样
```
T-shirt/top    : 0.0000
Trouser        : 0.0000
Pullover       : 0.0000
Dress          : 0.0000
Coat           : 0.0000
Sandal         : 0.3137
Shirt          : 0.0000
Sneaker        : 0.6370
Bag            : 0.0013
Ankle boot     : 0.2060
Sum: 1.1580
------------------------------
```
可以看到有一个输出总和超过了1

然后换成 `Softmax` , 总和就不会超过1了
```
------------------------------
T-shirt/top    : 0.1562
Trouser        : 0.1179
Pullover       : 0.1230
Dress          : 0.1726
Coat           : 0.0957
Sandal         : 0.0401
Shirt          : 0.1795
Sneaker        : 0.0219
Bag            : 0.0729
Ankle boot     : 0.0202
Sum: 1.0000
------------------------------
T-shirt/top    : 0.0000
Trouser        : 0.0000
Pullover       : 0.0000
Dress          : 0.0000
Coat           : 0.0000
Sandal         : 0.9874
Shirt          : 0.0000
Sneaker        : 0.0107
Bag            : 0.0001
Ankle boot     : 0.0017
Sum: 1.0000
```
现在模型输出的仍然是概率，需要从多个类别的概率中选择一个最高值。argmax() 可以找到最大值对应的索引，再通过这个索引获取对应的类别标签。
```python
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
predicted = probabilities.argmax().item()
print(f"预测结果：{class_names[predicted]}")
```



