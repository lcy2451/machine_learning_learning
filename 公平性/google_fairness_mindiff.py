"""Google 机器学习公平性练习。"""

# 在这里跟随课程逐步添加练习代码。
# 如果练习需要本地数据集，请统一放在仓库根目录的 data 文件夹中。
# https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/fairness_income.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=fairness#scrollTo=Rqy7tAHmW6Pu
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import tensorflow as tf

import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.proto import config_pb2
from google.protobuf import text_format

from tensorflow_model_remediation import min_diff


_csv = str(Path(__file__).parent.parent / 'data' / 'acsincome_raw_2018.csv')

# Import the dataset
acs_df = pd.read_csv(filepath_or_buffer=_csv)

# 随机打印 Pandas DataFrame 中的 5 行数据。
# print(acs_df.sample(5))
# 本次任务是预测一名美国成年劳动者的年收入是否超过 50,000 美元。
# 数据集中的 PINCP（个人年度总收入）列代表目标变量（Target Variable）。但是，需要将该变量转换为二元标签（Binary Label）。
# 对于每个样本：
# - 如果 PINCP > 50000.0，目标标签为 1。
# - 否则，目标标签为 0。
# 这是二值化， 二值化和分箱很像
# 二值化可以看作分箱的一种特殊情况：只分成两组。
LABEL_KEY = 'PINCP'
LABEL_THRESHOLD = 50000.0

acs_df[LABEL_KEY] = acs_df[LABEL_KEY].apply(
    lambda income: 1 if income > LABEL_THRESHOLD else 0)

# print(acs_df.sample(10))

# 特征整理、归一化和神经网络模型构建。
inputs = {}
features = acs_df.copy()
features.pop(LABEL_KEY)

# Instantiate a Keras input node for each column in the dataset.
# 为数据集中的每一列实例化一个 Keras 输入节点。
for name, column in features.items():
  if name != LABEL_KEY:
    inputs[name] = tf.keras.Input(
        shape=(1,), name=name, dtype=tf.float64)

# Stack the inputs as a dictionary and preprocess them.
# 将字典中的输入特征堆叠起来，并对其进行预处理。
def stack_dict(inputs, fun=tf.stack):
  values = []
  for key in sorted(inputs.keys()):
    values.append(tf.cast(inputs[key], tf.float64))

  return fun(values, axis=-1)

x = stack_dict(inputs, fun=tf.concat)

# Collect the features from the DataFrame, stack them together and normalize
# their values by passing them to the normalization layer.
# 从 DataFrame 中提取特征，将它们堆叠在一起，然后通过归一化层（Normalization Layer）对特征值进行归一化处理。
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(stack_dict(dict(features)))

# Build the main body of the model using a normalization layer, two dense
# rectified-linear layers, and a single output node for classification.
# 使用一个归一化层（Normalization Layer）、两个采用修正线性单元（ReLU）激活函数的全连接层（Dense Layers），以及一个用于分类的输出节点，构建模型的主体结构。
x = normalizer(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Put it all together using the Keras Functional API
# 使用 Keras 函数式 API（Functional API），将所有部分组合起来，构建一个完整的模型。
base_model = tf.keras.Model(inputs, outputs)

# 由于这是一个二元分类任务，计算真实标签与预测标签之间的交叉熵损失对于本练习来说已经足够。
# Define the metrics used to monitor model performance while training.
# 定义用于在模型训练过程中监控模型性能的评估指标。
METRICS = [
  tf.keras.metrics.BinaryAccuracy(name='accuracy'),
  tf.keras.metrics.AUC(name='auc'),
]

# Configure the model for training using a stochastic gradient descent
# optimizer, cross-entropy loss between true labels and predicted labels, and
# the metrics defined above to evaluate the base model during training.
# 使用随机梯度下降优化器（Stochastic Gradient Descent Optimizer）、
# 真实标签与预测标签之间的交叉熵损失（Cross-Entropy Loss），
# 以及上面定义的评估指标（Metrics）来配置模型，以便在训练过程中评估基础模型的性能。
base_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=METRICS)

# Helper function to convert a pandas DataFrame into a tf.Data.dataset object
# necessary for the purposes of this exercise.
# 辅助函数：将 Pandas DataFrame 转换为 tf.data.Dataset 对象，这是完成本次练习所必需的。
def dataframe_to_dataset(dataframe):
  dataframe = dataframe.copy()
  labels = dataframe.pop(LABEL_KEY)
  dataset = tf.data.Dataset.from_tensor_slices(
      ((dict(dataframe), labels)))
  return dataset

# 最终确定训练集并训练基础模型
RANDOM_STATE = 200
BATCH_SIZE = 100
EPOCHS = 10

# Use the sample() method in pandas to split the dataset into a training set
# that represents 80% of the original dataset, then convert it to a
# tf.data.Dataset object, and finally train the model using the
# converted training set.
# 使用 Pandas 的 sample() 方法，将原始数据集随机抽取 80% 作为训练集（Training Set），然后将其转换为 tf.data.Dataset 对象，最后使用转换后的训练集训练模型。
acs_train_df = acs_df.sample(frac=0.8, random_state=RANDOM_STATE)
acs_train_ds = dataframe_to_dataset(acs_train_df)
acs_train_batches = acs_train_ds.batch(BATCH_SIZE)

base_model.fit(acs_train_batches, epochs=EPOCHS)

# 评估基础模型
# Use the indices from the training set to create the test set, which represents
# 20% of the original dataset; then convert it to a tf.data.Dataset object, and
# evaluate the base model using the converted test set.
# 使用训练集的索引（Indices），从原始数据集中创建占比 20% 的测试集（Test Set）；
# 然后将其转换为 tf.data.Dataset 对象，最后使用转换后的测试集评估基础模型的性能。
acs_test_df = acs_df.drop(acs_train_df.index).sample(frac=1.0)
acs_test_ds = dataframe_to_dataset(acs_test_df)
acs_test_batches = acs_test_ds.batch(BATCH_SIZE)

print("evaluate   ", base_model.evaluate(acs_test_batches, batch_size=BATCH_SIZE))

# 评估公平性
# Generate output predictions using the test set.
# 使用测试集生成模型的输出预测结果。
base_model_predictions = base_model.predict(
    acs_test_batches, batch_size=BATCH_SIZE)

print("预测结果-> ", base_model_predictions)

SENSITIVE_ATTRIBUTE_VALUES = {1.0: "Male", 2.0: "Female"}
SENSITIVE_ATTRIBUTE_KEY = 'SEX'
PREDICTION_KEY = 'PRED'

# Make a copy of the test set, replace sensitive attribute values with
# categorial strings (for ease of visualization), and add predictions
# from the test set to the copied DataFrame as a separate column.
base_model_analysis = acs_test_df.copy()
base_model_analysis[SENSITIVE_ATTRIBUTE_KEY] = (
    base_model_analysis[SENSITIVE_ATTRIBUTE_KEY].replace(
        SENSITIVE_ATTRIBUTE_VALUES
    )
)
base_model_analysis[PREDICTION_KEY] = base_model_predictions

# Show five random examples to ensure that it looks correct.
base_model_analysis.sample(5)

# 配置公平性指标
# Specify Fairness Indicators using eval_config.
eval_config_pbtxt = """
  model_specs {
    prediction_key: "%s"
    label_key: "%s"
  }
  metrics_specs {
    metrics { class_name: "ExampleCount" }
    metrics { class_name: "BinaryAccuracy" }
    metrics { class_name: "AUC" }
    metrics { class_name: "Precision" }
    metrics { class_name: "Recall" }
    metrics { class_name: "ConfusionMatrixPlot" }
  }
  slicing_specs {
    feature_keys: "%s"
  }
  slicing_specs {}
""" % (PREDICTION_KEY, LABEL_KEY, SENSITIVE_ATTRIBUTE_KEY)
eval_config = text_format.Parse(eval_config_pbtxt, config_pb2.EvalConfig())

# Run TensorFlow Model Analysis.
base_model_eval_result = tfma.analyze_raw_data(base_model_analysis, eval_config)
# print("base_model_eval_result -> ", base_model_eval_result)

# Render Fairness Indicators.
# tfma.addons.fairness.view.widget_view.render_fairness_indicator(
#     base_model_eval_result)


# 使用已有的测试集分析数据
analysis_df = base_model_analysis.copy()

# 真实标签、模型预测概率
analysis_df["prediction_probability"] = np.asarray(
    base_model_predictions
).reshape(-1)

groups = ["Male", "Female"]
colors = {"Male": "#4472C4", "Female": "#E69138"}
metrics = ["Accuracy 准确性", "Precision 精确率", "Recall 召回率", "FPR 假阳性率"]


def evaluate_group(group_df, threshold):
    y_true = group_df[LABEL_KEY].to_numpy(dtype=int)
    y_pred = (
        group_df["prediction_probability"].to_numpy()
        >= threshold
    ).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    def safe_div(a, b):
        return float(a / b) if b else 0.0

    return {
        "Accuracy 准确性": safe_div(tp + tn, tp + tn + fp + fn),
        "Precision 精确率": safe_div(tp, tp + fp),
        "Recall 召回率": safe_div(tp, tp + fn),
        "FPR 假阳性率": safe_div(fp, fp + tn),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


# 创建两行图表：指标对比 + 混淆矩阵
fig = make_subplots(
    rows=2,
    cols=2,
    specs=[
        [{"colspan": 2}, None],
        [{}, {}],
    ],
    subplot_titles=[
        "Fairness Metrics Comparison 公平性指标对比",
        "Male Confusion Matrix 男性混淆矩阵",
        "Female Confusion Matrix 女性混淆矩阵",
    ],
    vertical_spacing=0.18,
)

thresholds = np.round(np.arange(0.1, 0.91, 0.05), 2)

for index, threshold in enumerate(thresholds):
    results = {
        group: evaluate_group(
            analysis_df[
                analysis_df[SENSITIVE_ATTRIBUTE_KEY] == group
            ],
            threshold
        )
        for group in groups
    }

    visible = bool(np.isclose(threshold, 0.5))

    # 男性与女性指标对比
    for group in groups:
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=[results[group][m] for m in metrics],
                name=group,
                marker_color=colors[group],
                visible=visible,
                legendgroup=group,
                showlegend=(index == 8),
                hovertemplate=f"%{{x}}: %{{y:.2%}}<extra>{group}</extra>",
            ),
            row=1,
            col=1,
        )

    # 混淆矩阵
    for col, group in enumerate(groups, start=1):
        r = results[group]
        matrix = [
            [r["TN"], r["FP"]],
            [r["FN"], r["TP"]],
        ]

        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=["Predicted 0", "Predicted 1"],
                y=["Actual 0", "Actual 1"],
                text=matrix,
                texttemplate="%{text:,}",
                colorscale="Blues",
                showscale=False,
                visible=visible,
                hovertemplate="%{x}<br>%{y}<br>Count: %{z:,}<extra></extra>",
            ),
            row=2,
            col=col,
        )

# 每个阈值对应 4 条 trace
steps = []
for i, threshold in enumerate(thresholds):
    visibility = [False] * len(fig.data)
    for j in range(i * 4, i * 4 + 4):
        visibility[j] = True

    steps.append(
        dict(
            method="update",
            label=f"{threshold:.2f}",
            args=[
                {"visible": visibility},
                {"title": f"Fairness Analysis — Threshold: {threshold:.2f}"},
            ],
        )
    )

fig.update_layout(
    title="Fairness Analysis — Threshold: 0.50",
    height=850,
    barmode="group",
    template="plotly_white",
    sliders=[
        dict(
            active=8,
            currentvalue={"prefix": "Classification Threshold: "},
            pad={"t": 45},
            steps=steps,
        )
    ],
)

fig.update_yaxes(
    range=[0, 1],
    tickformat=".0%",
    row=1,
    col=1,
)

fig.write_html(str(Path(__file__).parent / "fairness_report.html"), auto_open=True)

# A pandas DataFrame offers many approaches when it comes to indexing and
# selecting rows. One approach is by using boolean indexing
# (e.g., df[df['col'] == value]) as demonstrated in the following code:
# 在索引（Indexing）和选择行（Selecting Rows）方面，Pandas DataFrame 提供了多种方法。
# 其中一种方法是使用布尔索引（Boolean Indexing），例如 df[df['col'] == value]，如下方代码所示
sensitive_group_pos = acs_train_df[
    (acs_train_df[SENSITIVE_ATTRIBUTE_KEY] == 2.0) & (acs_train_df[LABEL_KEY] == 1)]
non_sensitive_group_pos = acs_train_df[
    (acs_train_df[SENSITIVE_ATTRIBUTE_KEY] == 1.0) & (acs_train_df[LABEL_KEY] == 1)]

# To learn more, visit: https://pandas.pydata.org/docs/user_guide/indexing.html

print(len(sensitive_group_pos),
      'positively labeled sensitive group examples 被标记为正例的敏感组样本')
print(len(non_sensitive_group_pos),
      'positively labeled non-sensitive group examples 被标记为正例的非敏感组样本')

# 准备 MinDiff 数据

# Convert sensitive and non-sensitive subsets into tf.data.Dataset.
MIN_DIFF_BATCH_SIZE = 50
sensitive_group_ds = dataframe_to_dataset(sensitive_group_pos)
non_sensitive_group_ds = dataframe_to_dataset(non_sensitive_group_pos)

# Batch the subsets.
sensitive_group_batches = sensitive_group_ds.batch(
    MIN_DIFF_BATCH_SIZE, drop_remainder=True)
non_sensitive_group_batches = non_sensitive_group_ds.batch(
    MIN_DIFF_BATCH_SIZE, drop_remainder=True)

# 为 MinDiff 模型打包数据集
# All you have to do is include each of the three batched tf.data.Datasets into
# the arguments: the training set, the sensitive subset (female) and
# the non-sensitive subset (male).
acs_train_min_diff_ds = min_diff.keras.utils.pack_min_diff_data(
    original_dataset = acs_train_batches,
    sensitive_group_dataset = sensitive_group_batches,
    nonsensitive_group_dataset = non_sensitive_group_batches)

# 训练 MinDiff 模型
# Wrap the original model in a MinDiffModel.
min_diff_model = min_diff.keras.MinDiffModel(
    original_model=base_model,
    loss=min_diff.losses.MMDLoss(),
    loss_weight=1)

# Compile the model after wrapping the original model.
min_diff_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=METRICS)

# 大约需要 5 分钟。
# Train MinDiff model using the packed dataset instead of the training set.
min_diff_model.fit(acs_train_min_diff_ds, epochs=EPOCHS)

# 评估 MinDiff 模型
min_diff_model.evaluate(acs_test_batches, batch_size=BATCH_SIZE)

# Generate MinDiff output predictions using the test set.
min_diff_model_predictions = min_diff_model.predict(
    acs_test_batches, batch_size=BATCH_SIZE)

# Make a copy of the test set, replace attribute with categorical values,
# and add the MinDiff test set predictions to the copied DataFrame as a separate
# column.
min_diff_model_analysis = acs_test_df.copy()
min_diff_model_analysis[SENSITIVE_ATTRIBUTE_KEY].replace(
    SENSITIVE_ATTRIBUTE_VALUES, inplace=True)
min_diff_model_analysis[PREDICTION_KEY] = min_diff_model_predictions

# Run TensorFlow Model Analysis on the MinDiff model.
min_diff_model_eval_result = tfma.analyze_raw_data(
    min_diff_model_analysis, eval_config)

# 当前 TFMA 版本不支持旧的 Fairness Indicators 可视化接口
# tfma.addons.fairness.view.widget_view.render_fairness_indicator(
#     min_diff_model_eval_result
# )
# MinDiff 模型的 Plotly 公平性分析
analysis_df = min_diff_model_analysis.copy()

# 性别转换为可读名称，避免 chained assignment
analysis_df[SENSITIVE_ATTRIBUTE_KEY] = (
    analysis_df[SENSITIVE_ATTRIBUTE_KEY].replace(
        SENSITIVE_ATTRIBUTE_VALUES
    )
)

# 保存预测概率
analysis_df["prediction_probability"] = np.asarray(
    min_diff_model_predictions
).reshape(-1)

# 分类阈值
thresholds = np.round(np.arange(0.1, 0.91, 0.05), 2)

groups = ["Male", "Female"]
colors = {"Male": "#4472C4", "Female": "#E69138"}

metrics = [
    "Accuracy 准确性",
    "Precision 精确率",
    "Recall 召回率",
    "FPR 假阳性率",
]

# 创建图表
fig = make_subplots(
    rows=1,
    cols=1,
    subplot_titles=["MinDiff 公平性指标对比"],
)

for index, threshold in enumerate(thresholds):
    for group in groups:
        group_df = analysis_df[
            analysis_df[SENSITIVE_ATTRIBUTE_KEY] == group
        ]

        result = evaluate_group(group_df, threshold)

        fig.add_trace(
            go.Bar(
                x=metrics,
                y=[result[m] for m in metrics],
                name=group,
                marker_color=colors[group],
                visible=np.isclose(threshold, 0.5),
                showlegend=(index == 0),
                hovertemplate=(
                    "%{x}: %{y:.2%}<extra>" + group + "</extra>"
                ),
            )
        )

# 创建阈值滑块
steps = []

for i, threshold in enumerate(thresholds):
    visibility = [False] * len(fig.data)
    visibility[2 * i] = True
    visibility[2 * i + 1] = True

    steps.append({
        "method": "update",
        "label": f"{threshold:.2f}",
        "args": [
            {"visible": visibility},
            {"title": f"MinDiff Fairness Analysis — Threshold: {threshold:.2f}"},
        ],
    })

fig.update_layout(
    title="MinDiff Fairness Analysis — Threshold: 0.50",
    template="plotly_white",
    barmode="group",
    height=550,
    yaxis={
        "title": "Score",
        "range": [0, 1],
        "tickformat": ".0%",
    },
    sliders=[{
        "active": 8,
        "currentvalue": {"prefix": "Classification Threshold: "},
        "steps": steps,
    }],
)

# 导出独立报告
report_path = Path(__file__).parent / "min_diff_fairness_report.html"
fig.write_html(str(report_path), auto_open=True)

print(f"MinDiff 公平性报告已生成：{report_path}")
