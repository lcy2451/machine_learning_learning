# 数据集——训练集、验证集和测试集
所有优秀的软件工程项目都会投入大量精力来测试其应用程序。同样，我们强烈建议对机器学习模型进行测试，以评估其预测结果的准确性。
# 训练集、验证集和测试集
您应该使用一组不同于训练样本的数据来测试模型。正如稍后将介绍的，与使用训练时的相同样本进行测试相比，使用不同样本进行测试，更能证明模型的适用性。
那么，这些不同的样本从哪里来呢？
传统上，在机器学习中，我们通过拆分原始数据集来获取这些样本。因此，您可能会认为应该将原始数据集拆分为两个子集：
- 模型在其上进行训练的训练集（Training set）。
- 用于评估已训练模型的测试集（Test set）。

![测试集和训练集.png](%E6%B5%8B%E8%AF%95%E9%9B%86%E5%92%8C%E8%AE%AD%E7%BB%83%E9%9B%86.png)

将数据集划分为两个集合是一个不错的想法，但更好的方法是将数据集划分为三个子集。除了训练集和测试集之外，第三个子集是：
- 验证集 validation set 
![测试集和训练集.png](%E6%B5%8B%E8%AF%95%E9%9B%86%E5%92%8C%E8%AE%AD%E7%BB%83%E9%9B%86.png)
图中是更好的划分方式。

使用验证集来评估训练集的结果。在反复使用验证集表明你的模型能够做出良好预测之后，再使用测试集对模型进行双重检查。

下图展示了这一工作流程。图中，“调整模型”指的是对模型进行任何修改——从更改学习率，到添加或删除特征，再到从头设计一个全新的模型。
![一个良好的开发与测试工作流程.png](%E4%B8%80%E4%B8%AA%E8%89%AF%E5%A5%BD%E7%9A%84%E5%BC%80%E5%8F%91%E4%B8%8E%E6%B5%8B%E8%AF%95%E5%B7%A5%E4%BD%9C%E6%B5%81%E7%A8%8B.png)

>注意：当你在训练集中转换某个特征时，必须在验证集、测试集和真实世界数据集中进行相同的转换。

测试集和验证集会因反复使用而“磨损”。也就是说，你越是使用相同的数据来做出关于超参数设置或其他模型改进的决策，你对模型能在新数据上做出良好预测的信心就越低。因此，收集更多数据来“刷新”测试集和验证集是个好主意。重新开始是一次很好的重置。

---

## 测试集的其他问题
重复样本会影响模型评估。将数据集划分为训练集、验证集和测试集后，请删除验证集或测试集中与训练集样本重复的所有样本。只有使用全新的样本，而非重复样本，才能对模型进行公平的评估。
例如，假设有一个模型，使用主题行、邮件正文和发件人电子邮件地址作为特征，来预测某封电子邮件是否为垃圾邮件。假设您将数据按 80-20 的比例划分为训练集和测试集。训练后，模型在训练集和测试集上都达到了 99% 的精确率。您可能会预期测试集上的精确率更低，因此再次查看数据后发现，测试集中的许多样本与训练集中的样本重复。问题在于，在划分数据之前，您没有从输入数据库中清除同一封垃圾邮件的重复条目。您无意中在部分测试数据上进行了训练。

总之，一个好的测试集或验证集需要满足以下所有标准：
- 样本规模足够大，能够产生具有统计显著性的测试结果。
- 能够代表整个数据集。换句话说，不应选择与训练集具有不同数据特征的测试集。
- 能够代表模型在实际业务应用中将遇到的真实数据。
- 不包含任何与训练集重复的样本。

----

# 代码部分
我使用`sklearn`库里的 `train_test_split` 来划分训练集、验证集和测试集
[知乎](https://zhuanlan.zhihu.com/p/248634166) --- 
[官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split)

数据集使用 [葡萄酒质量](https://archive.ics.uci.edu/dataset/186/wine+quality)

训练文件在 [wine_split.py](wine_split.py)

`train_size` 如果是浮点的话就表示要包含在训练拆分中的数据集比例, 如果是整数，表示的是具体样本数

`test_size` 和`train_size`差不多， 如果设置为None， 则为`train_size`的补集

`random_state` 表示随机状态， 如果要每次训练划分的都一样，那随便给一个整数就行了， 让划分结果可复现，方便后续对比不同实验结果。

`shuffle` 表示是否在划分之前对数据进行洗牌。如果 shuffle=False，则 stratify 必须为 None。当shuffle=True时，random_state才起作用

`stratify`现阶段对我还没有用, 就不研究了

下面演示划分`X`和 `Y`的划分训练集和测试集， 但是用了两个`train_test_split`， 显得有点臃肿， 但也没有错
```python
from sklearn.model_selection import train_test_split
# 划分训练集和测试集
x_train, x_test = train_test_split(X, train_size=0.8, test_size=0.2, random_state=42, shuffle=True)
y_train, y_test = train_test_split(y, train_size=0.8, test_size=0.2, random_state=42, shuffle=True)
```
可以合并成一个
```python
from sklearn.model_selection import train_test_split
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42, shuffle=True)
```

Google的教程也说了， 使用训练集、验证集和测试集三个才是比较好的划分方式
这里我按 训练集 80 %， 验证集和测试集分别 10 % 的比例来划分的
```python
from sklearn.model_selection import train_test_split

# 划分训练集和验证集以及测试集
x_train, x_temp, y_train, y_temp = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42, shuffle=True)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, train_size=0.5, random_state=42, shuffle=True)
```

