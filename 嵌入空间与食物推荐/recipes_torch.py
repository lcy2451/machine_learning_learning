from typing import Any
import random
from pathlib import Path
from collections.abc import Callable
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
import torch
from fontTools.misc.cython import returns
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from early_stopping_pytorch import EarlyStopping


class FoodClassifier(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.embedding = torch.nn.Embedding(
            num_embeddings=2420,
            embedding_dim=3
        )
        self.network = torch.nn.Sequential(

            torch.nn.Linear(in_features=3, out_features=8),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=8, out_features=2420),
        )

    def forward(self, x):
        # 前向传播
        x = self.embedding(x)
        x = x.mean(dim=1)
        return self.network(x)

def train(
    train_load: DataLoader,
    model: torch.nn.Module,
    optimizer_muon: torch.optim.Optimizer,
    optimizer_adamw: torch.optim.Optimizer,
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
        optimizer_adamw.zero_grad()  # 清空之前的梯度
        optimizer_muon.zero_grad()  # 清空之前的梯度

        loss.backward()  # 计算梯度

        optimizer_adamw.step()  # 更新模型参数
        optimizer_muon.step()  # 更新模型参数

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

if __name__ == "__main__":
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(42)
    torch.manual_seed(42)

    # 训练轮数
    _epochs = 500
    # 批量大小
    _batch_size = 64
    # 学习率
    _learning_rate = 0.001

    # 早停耐心值
    _patience = 20

    # 正则化率
    _weight_decay = 0.1


    RAW_interactions = Path(__file__).parent.parent / "data" / "RAW_interactions.csv"
    RAW_interactions_pd = pd.read_csv(RAW_interactions)

    RAW_recipes = Path(__file__).parent.parent / "data" / "RAW_recipes.csv"
    RAW_recipes_pd = pd.read_csv(RAW_recipes)

    # # 清理无效评论
    # _drop_id = []
    # for i in range(RAW_interactions_pd.shape[0]):
    #     user_id = RAW_interactions_pd['user_id'][i]
    #     recipe_id = RAW_interactions_pd['recipe_id'][i]
    #     rating = RAW_interactions_pd['rating'][i]
    #     recipes_name:pd.Series = RAW_recipes_pd.loc[RAW_recipes_pd['id'] == recipe_id, 'name']
    #     if recipes_name.empty:
    #         _drop_id.append(i)
    # RAW_interactions_pd = RAW_interactions_pd.drop(index = _drop_id)
    # RAW_interactions_pd.to_csv(Path(__file__).parent.parent / "data" / "RAW_interactions_cleaned.csv", index=False)
    # print(f"{RAW_interactions_pd.shape}") # (234428, 5)

    # RAW_interactions_cleaned = Path(__file__).parent.parent / "data" / "RAW_interactions_cleaned.csv"
    # RAW_interactions_cleaned_pd = pd.read_csv(RAW_interactions_cleaned)

    # groupby("recipe_id")：按照菜谱ID进行分组。
    # ["user_id"]：获取每组中的用户ID。
    # nunique()：统计不同用户的数量，重复用户只计算一次。
    recipe_counts = RAW_interactions_pd.groupby("recipe_id")["user_id"].nunique()
    # 然后按照用户数量降序排列：
    # sort_values()根据Series的值进行排序。
    # - ascending = True：升序，从小到大。
    # - ascending = False：降序，从大到小。
    #  这个是为了打印方便
    recipe_counts = recipe_counts.sort_values(ascending=False)

    # 这行代码是为了获取至少被 30 个不同用户评价过的菜谱 ID。
    valid_recipes = recipe_counts[recipe_counts >= 30].index

    RAW_interactions_pd = RAW_interactions_pd[
        RAW_interactions_pd["recipe_id"].isin(valid_recipes)
    ].reset_index(drop=True)

    user_counts = RAW_interactions_pd.groupby("user_id")["recipe_id"].nunique()

    users = user_counts[user_counts >= 5].index.tolist()

    users_train, users_temp = train_test_split(
        users, train_size=0.8, random_state=42
    )

    users_val, users_test = train_test_split(
        users_temp, train_size=0.5, random_state=42
    )

    recipe_id_to_index = {}
    datasets = []

    def build_samples(userlist) -> tuple[torch.Tensor, torch.Tensor]:
        __x = []
        __y = []
        for user_id in userlist:
            recipes = RAW_interactions_pd.loc[
                RAW_interactions_pd["user_id"] == user_id,
                ["recipe_id", "rating"]
            ]
            recipes = recipes.sort_values("rating", ascending=False)
            recipes = recipes.drop_duplicates(subset="recipe_id").head(5)

            recipe_ids = recipes["recipe_id"].tolist()

            ___x = []
            for recipe_id in recipe_ids:
                if recipe_id not in recipe_id_to_index:
                    recipe_id_to_index[recipe_id] = len(recipe_id_to_index)

                ___x.append(recipe_id_to_index[recipe_id])
            # 每种菜谱轮流作为标签
            for i in range(len(___x)):
                __x.append(___x[:i] + ___x[i + 1:])
                __y.append(___x[i])

        _x = torch.tensor(__x, dtype=torch.long)
        _y = torch.tensor(__y, dtype=torch.long)

        return _x, _y


    # 分别获取三个数据集
    x_train, y_train = build_samples(users_train)
    x_val, y_val = build_samples(users_val)
    x_test, y_test = build_samples(users_test)

    # print(_x.shape)
    # print(_y.shape)
    # print(len(recipe_id_to_index))
    # torch.Size([3588, 3])
    # torch.Size([3588])
    # 13081
    # exit()

    # print(x_train.shape, y_train.shape)
    # print(x_val.shape, y_val.shape)
    # print(x_test.shape, y_test.shape)
    #
    # print("训练集标签类别数量:", y_train.unique().numel())
    #
    # print("验证集未知标签比例:",
    #       (~torch.isin(y_val, y_train)).float().mean().item())
    #
    # print("验证集未知输入菜谱比例:",
    #       (~torch.isin(x_val, x_train)).float().mean().item())

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    valid_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=_batch_size, shuffle=False)
    val_loader = DataLoader(valid_dataset, batch_size=_batch_size, shuffle=False)

    # exit()
    # 损失函数
    loss_fn = torch.nn.CrossEntropyLoss().to(_device)

    # 创建模型实例
    _model = FoodClassifier().to(_device)

    # 优化器
    muon_params = []
    adamw_params = []

    for name, param in _model.named_parameters():
        if name == "network.0.weight":
            muon_params.append(param)
        else:
            adamw_params.append(param)
        print(name)

    _optimizer_muon = torch.optim.Muon(
        muon_params,
        lr=_learning_rate,
        weight_decay=_weight_decay
    )

    _optimizer_adamw = torch.optim.AdamW(
        adamw_params,
        lr=_learning_rate,
        weight_decay=_weight_decay
    )

    _loss_on_validations = []
    _loss_on_trainings = []
    early_stopping = EarlyStopping(patience=_patience, verbose=True)
    for t in range(_epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss = train(
            train_load=train_loader,
            model=_model, optimizer_adamw=_optimizer_adamw,
            optimizer_muon=_optimizer_muon,
            loss_fn=loss_fn)
        _loss_on_trainings.append(train_loss)

        test_loss = test(
            test_load=val_loader,
            model=_model,
            loss_fn=loss_fn)
        _loss_on_validations.append(test_loss)
        early_stopping(test_loss, _model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print("Done!")
    print("end")

    # 如果保存了最佳模型权重，在这里恢复

    # 获取嵌入层权重
    _model.eval()
    embedding_weights = _model.embedding.weight.detach().cpu()
    print(embedding_weights.shape)
    print(embedding_weights)

    plt.xlabel("Training iterations(time)")  # 设置横坐标的名称
    plt.ylabel("Loss")

    plt.plot(_loss_on_trainings, label="Training loss")
    plt.plot(_loss_on_validations, label="Validation loss")
    plt.legend()
    plt.show()
