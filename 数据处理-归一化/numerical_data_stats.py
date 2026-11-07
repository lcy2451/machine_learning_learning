# 打开数学统计练习
import pandas as pd

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

training_df = pd.read_csv(filepath_or_buffer="../data/california_housing_train.csv")

# 获取数据集的统计信息。

# 以下代码会返回 DataFrame 中数据的基本统计信息。

print(training_df.describe())

# @title Solution (run this code block to view) { display-mode: "form" }

print("""以下这些列中可能包含异常值（outliers）：
- total_rooms：房间总数
- total_bedrooms：卧室总数
- population：人口
- households：家庭户数
- median_income：收入中位数（也可能存在异常值）""")