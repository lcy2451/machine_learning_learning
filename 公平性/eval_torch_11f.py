"""加载已保存的 PyTorch 收入分类模型，进行评估（11 个输入特征）。"""
from pathlib import Path

import onnxruntime as ort
import pandas as pd
import torch

# 使用pip安装sklearn 用来分训练集， 验证集， 测试集 pip install -U scikit-learn
from sklearn.model_selection import train_test_split

def calculate_normalize(x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 每个特征单独计算平均值和标准差
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)

    # 防止标准差为 0
    std = std.clamp(min=1e-7)

    # 标准化
    return mean, std, (x - mean) / std

if __name__ == '__main__':

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型路径以当前脚本所在目录为基准，不受运行时工作目录影响。
    _onnx_path = model_path = Path(__file__).resolve().parent / "models" / "mmd_torch_11f.onnx"
    _onnx_path.parent.mkdir(parents=True, exist_ok=True)

    _csv = Path(__file__).parent.parent / 'data' / 'acsincome_raw_2018.csv'
    _acs_income_dataset = pd.read_csv(_csv)
    _LABEL_KEY = "PINCP"
    # PINCP 个人年总收入
    _training_df: pd.Series | pd.DataFrame = _acs_income_dataset.loc[
        :, (_LABEL_KEY, "AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P", "ST")
    ]

    # 任务是预测一位美国在职成年人的年收入是否超过50, 000美元。
    # 数据集中的 PINCP （个人年总收入）列代表目标变量；
    # 不过，该值需要转换为二值形式。对于每个样本，
    # 如果 PINCP > 50000.0 ，则个体的目标标签为1 ，否则为 0 。
    _training_df.loc[_training_df[_LABEL_KEY] <= 50000.0, _LABEL_KEY] = 0
    _training_df.loc[_training_df[_LABEL_KEY] > 50000.0, _LABEL_KEY] = 1

    # 划分 80训练集
    _train_df, _test_df = train_test_split(
        _training_df, test_size=0.2, random_state=42, stratify=_training_df[_LABEL_KEY])

    _x: torch.Tensor = torch.tensor(_train_df.drop(columns=[_LABEL_KEY]).to_numpy(),
                                    dtype=torch.float32).to(_device)

    _test_x: torch.Tensor =torch.tensor(
        _test_df.drop(columns=[_LABEL_KEY]).to_numpy(),dtype=torch.float32).to(_device)
    _test_y = torch.tensor(
        _test_df[_LABEL_KEY].to_numpy(),
        dtype=torch.float32
    ).reshape(-1, 1).to(_device)

    _mean, _std, _normalize = calculate_normalize(_x)
    test_normalize = (_test_x - _mean) / _std

    session = ort.InferenceSession(str(_onnx_path), providers=["CPUExecutionProvider"])

    # | 计数 | 真实类别 | 预测类别 |
    # | --- | --- | --- |
    # | TP | 高收入 | 高收入 |
    # | TN | 非高收入 | 非高收入 |
    # | FP | 非高收入 | 高收入 |
    # | FN | 高收入 | 非高收入 |
    counts = {
        1: {"TP": 0, "TN": 0, "FP": 0, "FN": 0},  # 男性
        2: {"TP": 0, "TN": 0, "FP": 0, "FN": 0},  # 女性
    }
    t = f = 0
    number_of_people = len(test_normalize)
    for i, (_test, _label) in enumerate(zip(test_normalize[:number_of_people], _test_y)):

        label = int(_test_y[i].item())

        sex = int(_test_df["SEX"].iloc[i])

        onnx_logits = session.run(
            ["logits"],  # 要获取名为 logits 的输出 这是模型导出时定义的
            {"features":  # 模型输入的名称
                 _test.unsqueeze(0).detach().cpu().numpy()}
        )[0]

        # logits 转为预测概率
        probability = torch.from_numpy(onnx_logits).sigmoid()
        predicted = int(probability.item() >= 0.5)

        result = "正确" if predicted == label else "错误"

        # print(f"x {_test} label {label} SEX {sex} {result}")

        if probability.item() >= 0.5 and predicted == label:
            counts[sex]["TP"] += 1  # 真正例
            t +=1

        elif probability.item() < 0.5 and predicted == label:
            counts[sex]["TN"] += 1  # 真负例
            t += 1

        elif probability.item() >= 0.5 and label < 1:
            counts[sex]["FP"] += 1  # 假正例
            f+=1

        elif probability.item() < 0.5 and label >= 1:
            counts[sex]["FN"] += 1  # 假负例
            f += 1

        # print(i)
    #
    # print(counts)
    #
    # print(t, f)

    for sex, group in counts.items():
        tp = group["TP"]
        tn = group["TN"]

        fp = group["FP"]
        fn = group["FN"]

        total = tp + tn + fp + fn  # 总计

        accuracy = (tp + tn) / total  # 准确率
        precision = tp / (tp + fp) # 精确率
        recall = tp / (tp + fn) # 召回率
        FPR = fp / (fp + tn) # 假阳性率
        print("-"*50)
        print(f" {'男性' if sex == 1 else '女性'}")
        print(f"准确率  {accuracy}")
        print(f"精确率  {precision}")
        print(f"召回率  {recall}")
        print(f"假阳性率  {FPR}")
        print("-" * 50)
