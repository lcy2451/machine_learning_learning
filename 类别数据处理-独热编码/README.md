# 类别数据处理-独热编码

---

## One-hot encoding  独热编码

因为大多数机器学习算法只能接受数值型的输入数据，无法直接处理离散型的特征。为了解决这个问题，独热编码被引入。

https://www.runoob.com/pandas/pandas-pd-get-dummies.html

`pd.get_dummies()` 是 Pandas 库的顶级函数，用于将分类变量转换为独热编码格式。

本次练习使用 Bike Sharing 数据集， 用season预测cnt。
训练文件：[onehot_train.py](onehot_train.py)

[Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+datase)
https://zhuanlan.zhihu.com/p/686452650


---
用OneHotEncoder独热编码
导入所需要的库创建

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
```

转换为独热编码， 先用训练集学习规则， 然后用训练集和验证集还有测试集转换
也可以用 fit_transform 同时完成学习规则和转换， 分开做是要考虑 验证集还有测试集不用重复转换
```python
# 初始化 OneHotEncoder 对象
encoder = OneHotEncoder()

# X["season"]:pd.Series
# 学习规则
encoder.fit(X[["season"]])

# encoder.transform()
# print(encoder.categories_)
# 转换，不重新学习规则
df_transformed = encoder.transform(X[["season"]])
```

打印OneHotEncoder 对象的一些信息
```python
# 获取独热编码后的特征的分类信息
encoder.categories_

# 获取独热编码后的特征的名称
encoder.get_feature_names_out()
```

转换为tensor对象
```python
x = torch.as_tensor(df_transformed.toarray(), dtype=torch.float32)
```

这时创建模型需要有4个输入
```python
torch.nn.Linear(in_features=4, out_features=1)
```

但我觉得只靠season无法拟合我的模型， 我想增加更多的输入， 我可以这样处理
```python
combined = np.concatenate(
    [df_transformed.toarray(), X[["temp", "windspeed", "hum", "atemp", "temp", "weekday"]].values],
    axis=1
    )
x = torch.as_tensor(combined, dtype=torch.float32)
```
