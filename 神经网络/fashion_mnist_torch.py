import os
# 本机 OpenMP 重复加载问题的临时绕过设置，并未消除底层运行库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from typing import Any
from collections.abc import Callable

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from early_stopping_pytorch import EarlyStopping

class FashionClassifier(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.network = torch.nn.Sequential(
            torch.nn.Flatten(),

            torch.nn.Linear(in_features=784, out_features=256),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=256, out_features=128),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=128, out_features=10),
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
        batch_x = batch_x.to(_device)
        batch_y = batch_y.to(_device)

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
            X = X.to(_device)
            y = y.to(_device)
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
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 训练轮数
    _epochs = 500
    # 批量大小
    _batch_size = 64
    # 学习率
    _learning_rate = 0.001

    # 早停耐心值
    _patience = 20


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
    loss_fn = torch.nn.CrossEntropyLoss().to(_device)

    # 创建模型实例
    _model = FashionClassifier().to(_device)

    # 优化器
    _optimizer = torch.optim.AdamW(
        _model.parameters(),  # 需要优化的模型参数
        lr=_learning_rate,  # 学习率
        betas=(0.9, 0.999),  # 一阶与二阶动量的衰减系数
        eps=1e-8,  # 防止除零的小常数
        weight_decay=0.2  # 权重衰减（AdamW为解耦方式）
    )

    print(f"训练设备: {_device}")

    if _device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(_device)}")
        torch.cuda.synchronize(_device)

    start = time.perf_counter()

    early_stopping = EarlyStopping(patience=_patience, verbose=True)
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

        early_stopping(test_loss, _model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print("Done!")
    print("end")
    if _device.type == "cuda":
        torch.cuda.synchronize(_device)

    elapsed = time.perf_counter() - start

    print(f"训练总耗时: {elapsed:.2f} 秒")

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
            X = X.to(_device)
            y = y.to(_device)
            pred = _model(X)
            pred = torch.nn.functional.softmax(pred, dim=1)

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
