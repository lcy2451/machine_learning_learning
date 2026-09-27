#@title Copyright 2023 Google LLC. Double-click for license information.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @title Load the imports

# Examine a dataset containing measurements derived from images of two species of Turkish rice.
# 检查一个数据集，其中包含源自两种土耳其大米图像的测量数据。
# Create a binary classifier to sort grains of rice into the two species.
# 创建一个二元分类器，将米粒分为这两个品种。

import keras
import ml_edu.experiment
import ml_edu.results
import numpy as np
import pandas as pd
import plotly.express as px

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

print("Ran the import statements.")

# @title Load the dataset
rice_dataset_raw = pd.read_csv("../data/Rice_Cammeo_Osmancik.csv")

# Read and provide statistics on the dataset.
rice_dataset = rice_dataset_raw[[
    'Area',
    'Perimeter',
    'Major_Axis_Length',
    'Minor_Axis_Length',
    'Eccentricity',
    'Convex_Area',
    'Extent',
    'Class',
]]

rice_dataset.describe()

# 米粒的最小和最大长度（长轴长度，以像素为单位）分别是多少？
# 这个数据集里的米粒尺寸是从图像里测出来的，所以长度单位用的是 px（pixel，像素），不是毫米。
print(
    f'最短的米粒长轴长度是  {rice_dataset.Major_Axis_Length.min():.1f}px,'
    f'最长的米粒长轴长度是  {rice_dataset.Major_Axis_Length.max():.1f}px.'
)

# 最小和最大米粒之间的面积范围是多少？
print(
    f'最小的米粒面积是 {rice_dataset.Area.min()}px²'
    f'最大的米粒面积是 {rice_dataset.Area.max()} px²'
)

# 最大米粒的周长与平均值相差多少个标准差（ std ）？
print(
    '周长最大的米粒，其周长为'
    f' {rice_dataset.Perimeter.max():.1f}px'
    f'它比平均值高出约'
    f' ~{(rice_dataset.Perimeter.max() - rice_dataset.Perimeter.mean())/rice_dataset.Perimeter.std():.1f} 个标准差。'
    f' 标准差为 ({rice_dataset.Perimeter.std():.1f})'
    f' 平均周长为  ({rice_dataset.Perimeter.mean():.1f}px).'
)
print(
    f'计算过程为：({rice_dataset.Perimeter.max():.1f} -'
    f' {rice_dataset.Perimeter.mean():.1f})/{rice_dataset.Perimeter.std():.1f} ='
    f' {(rice_dataset.Perimeter.max() - rice_dataset.Perimeter.mean())/rice_dataset.Perimeter.std():.1f}'
)


# Z-score 标准化，
# 把每个数值特征都转换成“离平均值多少个标准差”。
# 计算每个数值列的平均值。
feature_mean = rice_dataset.mean(numeric_only=True)

# 计算每个数值列的标准差。
feature_std = rice_dataset.std(numeric_only=True)

# 找出所有数值类型的列
numerical_features = rice_dataset.select_dtypes('number').columns

# 对所有数值特征进行 Z-score 标准化
normalized_dataset = (
    rice_dataset[numerical_features] - feature_mean
) / feature_std

# Copy the class to the new dataframe
# 把 Class 类别列复制回来，因为类别不是数值特征，不需要做标准化。
normalized_dataset['Class'] = rice_dataset['Class']

# 标准化训练集的一些值。注意到大多数
#  Z分数落在-2和+2之间。
# 看标准化后的前 5 行。
# # print(normalized_dataset.head())

# 设置随机种子
keras.utils.set_random_seed(42)

# 为了训练模型，我们将任意地将 Cammeo 品种标记为“1”，将 Osmancik 品种标记为“0”。
normalized_dataset['Class_Bool'] = (
    # 如果 Class 是 Cammeo，结果为 True
    # 如果 Class 是 Osmancik，结果为 False
    # 根据下面的条件来设置 0或者1
    normalized_dataset['Class'] == 'Cammeo'
).astype(int)

# 展示随机选择的10行
# print(normalized_dataset.sample(10))


# 我们可以对数据集进行随机化处理，并将其划分为训练集、验证集和测试集，分别占数据集的 80%、10%和 10%。
# 计算 80% 和 90% 的分割位置
number_samples = len(normalized_dataset)
index_80th = round(number_samples * 0.8)
index_90th = index_80th + round(number_samples * 0.1)

# Randomize order and split into train, validation, and test with a .8, .1, .1 split

# 随机打乱数据集
shuffled_dataset = normalized_dataset.sample(frac=1, random_state=100)

# 前 80%：训练集
train_data = shuffled_dataset.iloc[0:index_80th]

# 80% ~ 90%：验证集
validation_data = shuffled_dataset.iloc[index_80th:index_90th]

# 90% ~ 100%：测试集
test_data = shuffled_dataset.iloc[index_90th:]

# 显示测试集的前 5 行
# print(test_data.head())

# 训练时，不能把标签列作为模型输入，
# 否则模型相当于提前看到了答案，这叫标签泄漏（label leakage）。
label_columns = ['Class', 'Class_Bool']

# 训练集：特征
# drop 可以理解成“删除/丢弃”。
train_features = train_data.drop(columns=label_columns)

# 训练集：标签
train_labels = train_data['Class_Bool'].to_numpy()

# 验证集：特征和标签
validation_features = validation_data.drop(columns=label_columns)
validation_labels = validation_data['Class_Bool'].to_numpy()

# 测试集：特征和标签
test_features = test_data.drop(columns=label_columns)
test_labels = test_data['Class_Bool'].to_numpy()


# 训练模型 我们将在 Eccentricity 、 Major_Axis_Length, 和 Area 上训练一个模型。
# Name of the features we'll train our model on.
input_features = [
    'Eccentricity',
    'Major_Axis_Length',
    'Area',
]

def create_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
    """Create and compile a simple classification model."""

    # metrics 一组 Keras 评估指标
    # 根据 settings.input_features，批量创建模型输入层
    model_inputs = [
      keras.Input(name=feature, shape=(1,))
      for feature in settings.input_features
    ]
    # Use a Concatenate layer to assemble the different inputs into a single
    # tensor which will be given as input to the Dense layer.
    # For example: [input_1[0][0], input_2[0][0]]

    # 把前面那些分开的输入特征拼在一起
    concatenated_inputs = keras.layers.Concatenate()(model_inputs)

    # 在创建最后的输出层，而且因为用了 sigmoid，它很明显是在做二分类。
    # units = 1  是一个全连接层。
    # activation = 激活函数
    # sigmoid = S形函数，常用于二分类输出。
    model_output = keras.layers.Dense(
      units=1, name='dense_layer', activation=keras.activations.sigmoid
    )(concatenated_inputs)

    # 创建一个从输入到输出的完整神经网络模型。
    model = keras.Model(inputs=model_inputs, outputs=model_output)

    # 调用 compile 方法，把这些层配置成一个 Keras 可以执行的模型。
    # 注意：分类任务使用的损失函数和回归任务使用的损失函数不同。
    model.compile(
      optimizer=keras.optimizers.RMSprop(
          settings.learning_rate
      ),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics,
    )
    return model


def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    labels: np.ndarray,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
    """Feed a dataset into the model in order to train it."""

    # keras.Model.fit 的 x 参数可以是一个数组列表，
    # 其中每个数组都存放一个特征的数据。
    features = {
      feature_name: np.array(dataset[feature_name])
      for feature_name in settings.input_features
    }

    history = model.fit(
      x=features,
      y=labels,
      batch_size=settings.batch_size,
      epochs=settings.number_epochs,
    )

    return ml_edu.experiment.Experiment(
      name=experiment_name,
      settings=settings,
      model=model,
      epochs=history.epoch,
      metrics_history=pd.DataFrame(history.history),
    )


print('Defined the create_model and train_model functions.')

# Let's define our first experiment settings.
settings = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.001,
    number_epochs=60,
    batch_size=100,
    classification_threshold=0.35,
    input_features=input_features,
)

metrics = [
    keras.metrics.BinaryAccuracy(
        name='accuracy', threshold=settings.classification_threshold
    ),
    keras.metrics.Precision(
        name='precision', thresholds=settings.classification_threshold
    ),
    keras.metrics.Recall(
        name='recall', thresholds=settings.classification_threshold
    ),
    keras.metrics.AUC(num_thresholds=100, name='auc'),
]

# Establish the model's topography.
model = create_model(settings, metrics)

# Train the model on the training set.
experiment = train_model(
    'baseline', model, train_features, train_labels, settings
)

# Plot metrics vs. epochs
ml_edu.results.plot_experiment_metrics(experiment, ['accuracy', 'precision', 'recall'])
ml_edu.results.plot_experiment_metrics(experiment, ['auc'])


def compare_train_validation(experiment: ml_edu.experiment.Experiment, validation_metrics: dict[str, float]):
    print('比较训练集和验证集的指标:')
    for metric, validation_value in validation_metrics.items():
        print('------')
        print(f'训练集 {metric}: {experiment.get_final_metric_value(metric):.4f}')
        print(f'验证集 {metric}:  {validation_value:.4f}')


# Evaluate validation metrics
validation_metrics = experiment.evaluate(validation_features, validation_labels)
compare_train_validation(experiment, validation_metrics)

