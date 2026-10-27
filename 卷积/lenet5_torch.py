from collections.abc import Callable

from pathlib import Path
import time
import functools

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader
# 使用 TorchMetrics 统计二分类准确率
from torchmetrics.classification import MulticlassAccuracy
import torch

def monitor_training(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        model = kwargs["model"]
        device = next(model.parameters()).device

        print(f"训练设备: {device}")

        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(device)}")
            torch.cuda.synchronize(device)

        start = time.perf_counter()

        result = func(*args, **kwargs)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - start

        print(f"训练总耗时: {elapsed:.2f} 秒")

        return result

    return wrapper

@monitor_training
def train(train_load:DataLoader, test_load:DataLoader,
          model:torch.nn.Module,
          optimizer:torch.optim.Optimizer,
          loss_fn:Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
          accuracy:MulticlassAccuracy,
          epochs:int,
          device:torch.device):

    for epoch in range(epochs):
        model.train()

        for batch_x, batch_y in train_load:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
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
            for val_x_batch, val_y_batch in test_load:
                # 在测试集上评估，不是独立验证集
                predictions = model(val_x_batch)

                batch_loss = loss_fn(predictions, val_y_batch)
                epoch_loss += batch_loss * len(val_x_batch)

                # 更新指标
                targets = val_y_batch.int().flatten()
                accuracy.update(predictions, targets)

        epoch_loss = epoch_loss / len(test_load.dataset)

        epoch_accuracy = accuracy.compute()
        print(f"epoch: {epoch + 1}/{epochs}"
              f" test loss: {epoch_loss:.4f}"
              f" accuracy: {epoch_accuracy:.4f}")

if __name__ == "__main__":
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    # 随机种子， 42 是“生命、宇宙以及一切的终极答案”。
    torch.manual_seed(42)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=ToTensor()
    )
    X = train_data.data.float().unsqueeze(1) / 255.0
    Y = train_data.targets.long()

    X = X.to(_device)
    Y = Y.to(_device)


    Test_X = test_data.data.float().unsqueeze(1) / 255.0
    Test_Y = test_data.targets.long()

    Test_X = Test_X.to(_device)
    Test_Y = Test_Y.to(_device)

    # 学习率
    _learning_rate = 0.05
    _batch_size = 4096
    _epochs = 5000

    _model = torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=0),
        torch.nn.ReLU(),

        torch.nn.AvgPool2d(kernel_size=2, stride=2),

        torch.nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=0),
        torch.nn.ReLU(),

        torch.nn.AvgPool2d(kernel_size=2, stride=2),

        torch.nn.Flatten(),

        torch.nn.Linear(in_features=16 * 5 * 5, out_features=128),
        torch.nn.ReLU(),

        torch.nn.Linear(in_features=128, out_features=64),
        torch.nn.ReLU(),

        torch.nn.Linear(64, 10),  # MNIST 十分类

    )

    _model = _model.to(_device)

    # 创建优化器
    _optimizer = torch.optim.SGD(
        _model.parameters(), lr=_learning_rate, weight_decay=1e-4, )

    # 定义指标
    _acc = MulticlassAccuracy(num_classes=10, average="micro").to(_device)

    # 训练数据集
    _dataset = TensorDataset(X, Y)
    _dataloader = DataLoader(_dataset, batch_size=_batch_size, shuffle=True)

    # 测试数据集
    _test_dataset = TensorDataset(Test_X, Test_Y)
    _test_dataloader = DataLoader(_test_dataset, batch_size=_batch_size, shuffle=True)

    # loss 使用categorical_crossentropy
    _loss_fn = torch.nn.CrossEntropyLoss()
    _loss_fn = _loss_fn.to(_device)

    train(train_load=_dataloader,
          test_load=_test_dataloader,
          model=_model,
          optimizer=_optimizer,
          loss_fn=_loss_fn,
          accuracy=_acc,
          epochs=_epochs,
          device=_device)
