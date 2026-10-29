# import os
# 本机 OpenMP 重复加载问题的临时绕过设置，并未消除底层运行库冲突
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from collections.abc import Callable
# 使用 TorchMetrics 统计二分类准确率
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

import torch
from torch.utils.data import DataLoader

def train(
    train_load: DataLoader,
    *,
    val_load: DataLoader | None = None,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    accuracy: Metric,
    epochs: int,
):
    if not val_load:
        val_load = train_load
    
    for epoch in range(epochs):

            # 模型切换到训练模式
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

                if not val_load:
                     val_load = train_load

                for val_x_batch, val_y_batch in val_load:
                     # 用验证集验证
                    predictions = model(val_x_batch)

                    batch_loss = loss_fn(predictions, val_y_batch)
                    epoch_loss += batch_loss.item()

                    # 更新指标
                    targets = val_y_batch.flatten()
                    accuracy.update(predictions.flatten(), targets)

            epoch_loss = epoch_loss / len(val_load.dataset)
            
            # 汇总整个验证集的指标
            epoch_accuracy = accuracy.compute()

            print(f"epoch: {epoch+1}/{epochs}, val loss: {epoch_loss:.4f},"
                  f" MAE: {epoch_accuracy:.4f}")

    weight = model.weight.item()
    bais = model.bias.item()

    print("weight: ", f"{weight:.4f}")
    print("bias: ", f"{bais:.4f}")
