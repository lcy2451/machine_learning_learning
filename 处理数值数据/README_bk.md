# 处理数值数据

本目录收录 Google 机器学习速成课程“处理数值数据”部分的两个编程练习，主要学习如何通过描述性统计、数据可视化和分组比较发现数值数据中的异常值。

参考课程：

- [数值数据：编程练习](https://developers.google.com/machine-learning/crash-course/numerical-data/programming-exercises?hl=zh-cn)
- [数学统计练习（Google Colab）](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/numerical_data_stats.ipynb?hl=zh-cn)
- [查找有缺陷数据的练习（Google Colab）](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/numerical_data_bad_values.ipynb?hl=zh-cn)

## 代码说明

| 文件 | 数据来源 | 主要内容 |
| --- | --- | --- |
| [numerical_data_stats.py](./numerical_data_stats.py) | [加利福尼亚州住房训练数据](../data/california_housing_train.csv) | 使用 pandas 查看各数值列的计数、平均值、标准差、四分位数、最小值和最大值，初步识别可能存在异常值的列 |
| [numerical_data_bad_values.py](./numerical_data_bad_values.py) | 文件内嵌的学生卡路里与考试成绩数据 | 按周、日期和学生定位数据，通过散点图函数与分组平均值检查隐藏的异常数据 |

## numerical_data_stats.py：通过统计信息检查异常值

脚本使用 `pandas.read_csv()` 读取加利福尼亚州住房训练数据，然后调用：

```python
training_df.describe()
```

`describe()` 会为每个数值列生成以下统计信息：

| 统计量 | 含义 |
| --- | --- |
| `count` | 非空数据的数量 |
| `mean` | 平均值 |
| `std` | 标准差，反映数据的离散程度 |
| `min` | 最小值 |
| `25%` | 第一四分位数 |
| `50%` | 中位数 |
| `75%` | 第三四分位数 |
| `max` | 最大值 |

当前脚本根据这些统计信息，提示以下列可能包含异常值：

- `total_rooms`：房间总数。
- `total_bedrooms`：卧室总数。
- `population`：人口。
- `households`：家庭户数。
- `median_income`：收入中位数。

仅凭统计信息可以发现数值范围特别大的列，但不能保证找出隐藏在某一部分数据中的错误值，因此第二个练习还会按数据位置进行检查。

## numerical_data_bad_values.py：查找隐藏的异常数据

内嵌数据包含两个字段：

| 字段 | 含义 |
| --- | --- |
| `calories` | 营养师记录的学生早餐卡路里数 |
| `test_score` | 学生当天的数学测试成绩 |

数据池包含 50 名学生，每名学生连续接受 28 天评估，因此共有：

```text
50 名学生 × 28 天 = 1400 条记录
```

数据按日期排列，每连续 50 行表示同一天的 50 名学生。一周包含 7 天，所以每周共有：

```text
50 名学生 × 7 天 = 350 条记录
```

代码使用下面的公式将周、日期和学生编号转换为 DataFrame 中的行号：

```python
position = (week * 350) + (day * 50) + subject
```

其中：

- `week * 350`：跳过前面完整周的数据。
- `day * 50`：跳过本周前面完整日期的数据。
- `subject`：定位到当天的某一名学生。

当前代码假设一周从星期一开始编号，因此 `day == 3` 表示星期四：

```text
0=星期一，1=星期二，2=星期三，3=星期四，
4=星期五，5=星期六，6=星期日
```

4 周中共有 4 个星期四，每天有 50 名学生，所以星期四共有：

```text
4 周 × 1 天 × 50 名学生 = 200 条记录
```

其他日期共有：

```text
4 周 × 6 天 × 50 名学生 = 1200 条记录
```

脚本使用实际累计的 `thursday_count` 和 `day_count` 作为除数计算平均值。当前运行结果为：

```text
星期四的平均卡路里值是 201
除星期四以外，其他天的平均卡路里值是 183
```

### 与原练习的日期编号差异

Google 原练习把 `Day 4` 称为 Thursday，参考答案使用 `day == 4`。按照 Python 从 0 开始的编号方式，`day == 4` 是每周第 5 组数据；原练习的数据中正是这组数据的卡路里范围异常，平均值约为 `93`，其他日期的平均值约为 `201`。

当前本地代码采用“星期一为 0”的编号方式，将 `day == 3` 解释为星期四。这样得到的是上方记录的 `201` 和 `183`，统计对象与原练习答案不同，也不会单独选中原数据中的异常组。README 记录的是当前代码的实际行为；如需复现原练习的结论，应按照原练习使用 `day == 4`。

文件还提供两个散点图函数：

- `plot_the_dataset()`：随机抽取指定数量的数据点，观察卡路里与测试成绩的整体关系。
- `plot_a_contiguous_portion_of_dataset()`：绘制指定连续区间的数据，用于比较不同日期的数据分布。

当前文件中的绘图调用处于注释状态，直接运行脚本时只会计算并输出两组平均卡路里值。

## 运行方式

在项目根目录打开终端，使用已有的 Anaconda `machine_learning` 环境：

```powershell
conda activate machine_learning
cd 处理数值数据
python numerical_data_stats.py
python numerical_data_bad_values.py
```

`numerical_data_stats.py` 使用相对路径 `../data/california_housing_train.csv`，因此需要从“处理数值数据”目录运行。两个脚本使用 pandas，异常数据练习还导入 Matplotlib 以绘制散点图。
