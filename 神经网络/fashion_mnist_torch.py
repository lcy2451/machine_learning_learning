from typing import Any
from collections.abc import Callable

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import torch

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
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 前向传播
        return self.network(x)

def train(
    train_load: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):

    size = len(train_load.dataset)
    model.train()

    test_loss = 0

    for batch, (batch_x, batch_y) in enumerate(train_load):

        # 1. 前向传播
        pred = model(batch_x)

        # 2. 计算 Loss
        loss = loss_fn(pred, batch_y)

        # 3. 反向传播
        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数

        if batch % 5 == 0 or True:
            loss, current = loss.item(), min((batch + 1) * train_load.batch_size, size)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            test_loss += loss
    test_loss /= len(train_load)
    return test_loss


def test(
    test_load: DataLoader,
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
):
    num_batches = len(test_load)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in test_load:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n , Avg loss: {test_loss:>8f} \n")

    return test_loss

def target_transform(y: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(
        torch.tensor(y, dtype=torch.int64),
        num_classes=10
    ).float()

if __name__ == "__main__":
    # 训练轮数
    _epochs = 500
    # 批量大小
    _batch_size = 256
    # 学习率
    _learning_rate = 0.001

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

    train_loader = DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=_batch_size, shuffle=False)

    # 损失函数
    loss_fn = torch.nn.BCELoss()

    # 创建模型实例
    _model = FashionClassifier()

    # 优化器
    _optimizer = torch.optim.AdamW(
        _model.parameters(),  # 需要优化的模型参数
        lr=_learning_rate,  # 学习率
        betas=(0.9, 0.999),  # 一阶与二阶动量的衰减系数
        eps=1e-8,  # 防止除零的小常数
                weight_decay=1e-2  # 权重衰减（AdamW为解耦方式）
    )

    _loss_on_validations = []
    _loss_on_trainings = []
    for t in range(_epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss = train(
            train_load=train_loader,
            model=_model, optimizer=_optimizer,
            loss_fn=loss_fn)
        _loss_on_trainings.append(train_loss)

        test_loss = test(
            test_load=test_loader,
            model=_model,
            loss_fn=loss_fn)
        _loss_on_validations.append(test_loss)

        print("Done!")
    print("end")

    plt.xlabel("Training iterations(time)")  # 设置横坐标的名称
    plt.ylabel("Loss")

    plt.title("MNIST")
    plt.plot(_loss_on_trainings, label="Training loss")
    plt.plot(_loss_on_validations, label="Validation loss")
    plt.legend()
    plt.show()

    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    _model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            pred = _model(X)

            for probabilities, _y in zip(pred, y):
                for name, p in zip(class_names, probabilities):
                    print(f"{name:<15}: {p.item():.4f}")

                predicted = probabilities.argmax().item()
                true_label = _y.argmax().item()

                print(f"Sum: {probabilities.sum().item():.4f}")
                print(f"预测结果：{class_names[predicted]}")
                print(f"真实结果  {class_names[true_label]}")
                print("-" * 30)


            break
