"""使用 MMD 分布差异损失进行公平性干预训练（PyTorch，11 个输入特征）。"""

# 基础训练文件：mlp_torch_11f.py。
# 样本加权训练文件：finetune_torch_11f.py。
# 分组评估文件：eval_torch_11f.py。
# 数据集：仓库根目录 data/acsincome_raw_2018.csv。
# 从选定的 .pt 检查点恢复模型参数、标准化参数和特征顺序。
# 使用原训练集训练，在原验证集上比较效果。
# 对真实高收入的男女两组预测分数计算 MMD，作为附加损失。
# 总损失 = 分类损失 + MMD 权重 × MMD 损失。
# MMD 权重控制分布差异惩罚的强度，与样本加权系数不同。
# 调整后的 .pt 和 ONNX 使用独立文件名，保留已有实验版本。
# 在这里逐步编写检查点加载、MMD 损失和训练流程。
import time
from pathlib import Path
import functools
import pandas as pd
from collections.abc import Callable

import torch
from torch.utils.data import TensorDataset, DataLoader

# 安装 pytorch 指标 python -m pip install torchmetrics
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

# 使用pip安装sklearn 用来分训练集， 验证集， 测试集 pip install -U scikit-learn
from sklearn.model_selection import train_test_split

def calculate_mmd(
    male_scores: torch.Tensor,
    female_scores: torch.Tensor,
    sigma: float = 0.2,
) -> torch.Tensor:
    """计算两组一维预测概率的高斯核 MMD²（有偏估计）。"""
    if sigma <= 0:
        raise ValueError("sigma 必须大于 0")

    # 某批次缺少其中一组时，不计算分布差异。
    if male_scores.numel() == 0 or female_scores.numel() == 0:
        return male_scores.new_zeros(())

    male = male_scores.reshape(-1, 1)
    female = female_scores.reshape(-1, 1)

    # 两两概率差的平方。
    distance_mm = (male - male.T).square()
    distance_ff = (female - female.T).square()
    distance_mf = (male - female.T).square()

    # 高斯核：距离越小，相似度越高。
    kernel_mm = torch.exp(-distance_mm / (2 * sigma**2))
    kernel_ff = torch.exp(-distance_ff / (2 * sigma**2))
    kernel_mf = torch.exp(-distance_mf / (2 * sigma**2))

    mmd = kernel_mm.mean() + kernel_ff.mean() - 2 * kernel_mf.mean()

    # 避免浮点误差产生极小的负数。
    return mmd.clamp_min(0)

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
def train(train_loader:DataLoader,
          val_loader:DataLoader,
          model:torch.nn.Module,
          loss_fn:Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
          optimizer:torch.optim.Optimizer,
          accuracy: BinaryAccuracy,
          auc: BinaryAUROC,
          epochs:int,
          mmd_weight = 0.1):

    # 开始按 Epoch 训练
    for epoch in range(epochs):
        # 模型切换到训练
        model.train()

        for x_batch, y_batch, sex_batch in train_loader:
            # 1. 前向传播
            _logits = model(x_batch)

            # 2. 计算 Loss

                # 分类损失
            classification_loss = loss_fn(_logits, y_batch)

            probabilities = _logits.sigmoid().flatten()
            labels = y_batch.flatten()
            sex = sex_batch.flatten()

            # 真实高收入男性的预测概率
            male_scores = probabilities[(sex == 1) & (labels == 1)]

            # 真实高收入女性的预测概率
            female_scores = probabilities[(sex == 2) & (labels == 1)]
            mmd_loss = calculate_mmd(male_scores, female_scores)

            loss = classification_loss + mmd_weight * mmd_loss

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
        auc.reset()

        # 关闭梯度记录
        with torch.no_grad():

            epoch_loss = 0
            for val_x_batch, val_y_batch, val_sex_batch in val_loader:
                # 用验证集验证
                predictions = model(val_x_batch)
                # 2. 计算 Loss

                    # 分类损失
                classification_loss = loss_fn(predictions, val_y_batch)

                epoch_loss += classification_loss.item() * len(val_x_batch)

                # 更新指标
                probabilities = predictions.sigmoid().flatten()
                targets = val_y_batch.int().flatten()
                accuracy.update(probabilities, targets)
                auc.update(probabilities, targets)

            epoch_loss = epoch_loss / len(val_loader.dataset)

            # 汇总整个验证集的指标
            epoch_accuracy = accuracy.compute()
            epoch_auc = auc.compute()

            print(f"epoch: {epoch+1}/{epochs}, val loss: {epoch_loss:.4f},"
                  f" accuracy: {epoch_accuracy:.4f}, auc: {epoch_auc:.4f}")

if __name__ == "__main__":
    _mmd_weight = 0.1
    # 学习率
    _learning_rate = 0.001
    _batch_size = 100
    _epochs = 10

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 加权模型的pt
    _pt_path = model_path = Path(__file__).resolve().parent / "models" / "mlp_torch_11f_weighted.pt"
    _csv = Path(__file__).parent.parent / 'data' / 'acsincome_raw_2018.csv'
    _acs_income_dataset = pd.read_csv(_csv)
    _LABEL_KEY = "PINCP"
    # PINCP 个人年总收入
    _training_df: pd.Series | pd.DataFrame = _acs_income_dataset.loc[
        :, (_LABEL_KEY, "AGEP",  "COW",  "SCHL",  "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P", "ST")
    ]

    # 任务是预测一位美国在职成年人的年收入是否超过50, 000美元。
    # 数据集中的 PINCP （个人年总收入）列代表目标变量；
    # 不过，该值需要转换为二值形式。对于每个样本，
    # 如果 PINCP > 50000.0 ，则个体的目标标签为1 ，否则为 0 。
    _training_df.loc[_training_df[_LABEL_KEY] <= 50000.0, _LABEL_KEY] = 0
    _training_df.loc[_training_df[_LABEL_KEY] > 50000.0, _LABEL_KEY] = 1

    _features = _training_df.drop(columns=["PINCP"])
    # _train_loader = DataLoader(_dataset, batch_size=_batch_size, shuffle=True)
    # 划分 80训练集
    _train_df, _temp_df = train_test_split(
        _training_df, test_size=0.2, random_state=42, stratify=_training_df[_LABEL_KEY])

    # 这次mmd训练要加上"SEX"
    _train_sex = torch.tensor(
        _train_df["SEX"].to_numpy(),
        dtype=torch.int64,
        device=_device,
    )

    # 划分验证集和测试集
    _val_df, _test_df = train_test_split(
        _temp_df, test_size=0.5, random_state=42, stratify=_temp_df[_LABEL_KEY]
    )

    # 这次mmd训练要加上"SEX"
    _val_sex = torch.tensor(
        _val_df["SEX"].to_numpy(),
        dtype=torch.int64,
        device=_device,
    )

    _x: torch.Tensor = torch.tensor(_train_df.drop(columns=[_LABEL_KEY]).to_numpy(),
                                    dtype=torch.float32).to(_device)
    # 真实标签 PINCP 真实答案
    _y = torch.tensor(_train_df[_LABEL_KEY].to_numpy(),dtype=torch.float32).reshape(-1, 1).to(_device)

    _val_x: torch.Tensor = torch.tensor(_val_df.drop(columns=[_LABEL_KEY]).to_numpy(),dtype=torch.float32).to(_device)
    _y_val = torch.tensor(
        _val_df[_LABEL_KEY].to_numpy(),
        dtype=torch.float32
    ).reshape(-1, 1).to(_device)

    checkpoint = torch.load(
        str(_pt_path),
        map_location=_device,
        weights_only=True
    )

    _model = torch.nn.Sequential(
        torch.nn.Linear(_x.shape[1], 64),
        torch.nn.ReLU(),

        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),

        torch.nn.Linear(32, 1),
        # torch.nn.Sigmoid(),
    )

    _model = _model.to(_device)

    _model.load_state_dict(checkpoint["model_state_dict"])

    # 创建优化器
    _optimizer = torch.optim.Adam(_model.parameters(), lr=_learning_rate)
    _optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 使用保存的标准化参数处理数据
    _mean = checkpoint["mean"].to(_device)
    _std = checkpoint["std"].to(_device)
    # 标准化 → 创建 Dataset、DataLoader
    _normalize = (_x - _mean) / _std
    val_normalize = (_val_x - _mean) / _std

    # 创建损失函数
    _loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

    # 定义指标
    _accuracy = BinaryAccuracy().to(_device)
    _auc = BinaryAUROC().to(_device)

    # 继续训练
    # 训练数据集 但是不加权
    _train_dataset = TensorDataset(_normalize, _y, _train_sex)
    _train_loader = DataLoader(_train_dataset, batch_size=_batch_size, shuffle=True)

    # 验证数据集
    _val_dataset = TensorDataset(val_normalize, _y_val, _val_sex)
    _val_loader = DataLoader(_val_dataset, batch_size=_batch_size, shuffle=False)

    train(train_loader=_train_loader, val_loader=_val_loader,
          model=_model, loss_fn=_loss_fn, optimizer=_optimizer,
          accuracy=_accuracy, auc=_auc, epochs=_epochs, mmd_weight=_mmd_weight)

    # 模型路径以当前脚本所在目录为基准，不受运行时工作目录影响。
    _onnx_path  = Path(__file__).resolve().parent / "models" / "mmd_torch_11f.onnx"
    _pt_path = _onnx_path.with_suffix(".pt")
    _onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        _model,
        (_normalize[:1],),
        _onnx_path,
        input_names=[ "features"],
        output_names=["logits"],
        dynamo=True,
        external_data=False

    )

    torch.save(
        {
            "model_state_dict": _model.state_dict(),
            "optimizer_state_dict": _optimizer.state_dict(),
            "mean": _mean.detach().cpu(),
            "std": _std.detach().cpu(),
            "feature_names": _train_df.drop(columns=[_LABEL_KEY]).columns.to_list(),
            "epochs": _epochs,
        },
        _pt_path,
    )
