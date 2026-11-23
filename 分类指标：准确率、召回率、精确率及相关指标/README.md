Accuracy准确率是指所有分类中判断正确的比例，无论正例还是负例。

在垃圾邮件分类示例中，准确率衡量的是所有电子邮件中被正确分类的比例。

一个完美的模型将具有零假正例和零假负例，因此准确率为 1.0，即 100%。

由于它包含了混淆矩阵中的全部四种结果（TP、FP、TN、FN），在数据集平衡、两个类别的样本数量相近的情况下，准确率可以作为模型质量的粗略度量。因此，它通常是执行通用或未指定任务的通用或未指定模型所使用的默认评估指标。

然而，当数据集不平衡，或者一种错误（FN 或 FP）的代价高于另一种错误时——这在大多数实际应用中都是如此——最好改为针对其他指标之一进行优化。

对于严重不平衡的数据集，即某一类别出现得非常罕见（例如仅占 1% 的时间），一个始终预测负例的模型在准确率上仍会得到 99% 的分数，尽管它毫无用处。

## 召回率，或真阳性率
真阳性率（TPR），即所有实际阳性中被正确分类为阳性的比例，也称为召回率（Recall）。
    
假阴性是实际为正类，却被模型错误分类为负类的样本。

在垃圾邮件分类示例中，召回率衡量的是实际垃圾邮件中，被模型正确识别为垃圾邮件的比例。

因此，召回率也被称为**检测概率（Probability of Detection）**，它回答的是：“模型成功检测出了多少垃圾邮件？”

一个假设的完美模型将具有零假阴性，因此召回率（TPR）为 1.0，也就是说，检测率为 100%。

在实际阳性数量非常少的不平衡数据集中，召回率是比准确率更有意义的指标。

# False positive rate假阳性率 或 Fall-out（误报率）
假阳性率（FPR）是所有实际阴性中被错误分类为阳性的比例，也称为误报概率。

假正例是被错误分类的实际负例，在垃圾邮件分类示例中，FPR 衡量的是被错误分类为垃圾邮件的合法邮件所占的比例，即模型的误报率。

一个完美的模型不会有假正例，因此 FPR 为 0.0，也就是说，误报率为 0%。

# Precision  精确率
精确率是模型所有正例分类中实际为正例的比例。

在垃圾邮件分类示例中，精确率（Precision）衡量的是所有被模型判定为垃圾邮件的邮件中，实际属于垃圾邮件的比例。

一个假设的完美模型不会有假正例，因此精确率为 1.0。

# 指标选择与权衡
在评估模型和选择阈值时，应该优先考虑哪些指标，取决于具体问题所涉及的成本、收益和风险。

在垃圾邮件分类示例中，我们可以优先考虑召回率，尽可能识别出所有垃圾邮件；也可以优先考虑精确率，确保被标记为垃圾邮件的邮件确实是垃圾邮件；还可以在满足最低准确率要求的基础上，尽可能平衡两者。

| Metric 指标 | Guidance 指导 |
|---|---|
| **Accuracy 准确率** | 对于平衡数据集，可用作模型训练进度/收敛情况的粗略指标。<br>对于模型性能，仅应与其他指标结合使用。<br>避免用于不平衡数据集，考虑使用其他指标。 |
| **Recall 召回率（True Positive Rate 真阳性率）** | 当假阴性的代价高于假阳性时使用。 |
| **False Positive Rate 假阳性率** | 当假阳性的代价高于假阴性时使用。 |
| **Precision 精确率** | 当正类预测的准确性非常重要时使用。 |

---
# 代码部分
## Accuracy 准确率
要实现准确率， 我打算先尝试自己造轮子然后在用现成的轮子
轮子的公式是
$$
Accuracy = \frac{正确分类的数量}{总样本数量}=\frac{TP+TN}{TP + TN + FP + FN}
$$
其中TP + TN 是 混淆矩阵中的真正例和真假例， 我可以创建一个混淆矩阵只要他的TP 和 TN

也就是这样
```python
from torchmetrics.classification import BinaryConfusionMatrix
correct = 0
confusion_matrix = BinaryConfusionMatrix(threshold=0.5)
result = confusion_matrix(predictions, y)
correct+= result[0][0].item() + result[1][1].item()
accuracy = correct / num_samples
```
## PyTorch Ignite

PyTorch Ignite 是一个基于 PyTorch 的高级训练与评估工具库（High-level Library）。它不是 PyTorch 本身，也不是另一个深度学习框架。它主要是帮你减少训练代码中的重复工作。

https://pytorch-ignite.ai/tutorials/beginner/01-getting-started/

安装 Ignite
```
pip install pytorch-ignite
```

```python
from ignite.engine import Engine
from ignite.metrics import Accuracy

def process_function(engine, batch):
    # ...
    X, y = batch
    logits =  model(X)
    y_pred = torch.sigmoid(logits)
    y_pred = torch.where(y_pred > 0.5, 1, 0)
    return y_pred, y
engine = Engine(process_function)
metric = Accuracy()
metric.attach(engine, "accuracy")
state = engine.run(test_load)
print(f"Accuracy: {state.metrics['accuracy']}")
```
Accuracy（准确率）主要用于分类任务。

# 召回率
召回率的实现也是手搓轮子和用现成的轮子

首先是手搓， 召回率的公式是：
$$
Recall = \frac{实际为正且被正确预测的样本数}{实际为正的样本总数} = \frac{TP}{TP+FN}
$$

```python
tp = 0
tp_fn = 0
tp += result[1][1].item()
tp_fn += result[1][1].item() + result[1][0].item()
recall = tp / tp_fn
```
召回率也可以用Ignite实现
```python
from ignite.engine import Engine
from ignite.metrics import  Recall
def process_function(engine, batch):
    # ...
    X, y = batch
    logits =  model(X)
    y_pred = torch.sigmoid(logits)
    y_pred = torch.where(y_pred > 0.5, 1, 0)
    return y_pred, y
engine = Engine(process_function)
metric_recall = Recall()
metric_recall.attach(engine, "recall")
state = engine.run(test_load)
print(f"Recall: {state.metrics['recall']}")
```
# 假阳性率
这个只看手搓轮子了， Ignite没有现成的
假阳性率的公式是：
$$
FPR = \frac{被错误分类的实际负样本数}{所有实际负样本数} = \frac{FP} {FP + TN}
$$
```python
fp = 0
fp_tn = 0
fp += result[0][1].item()
fp_tn += result[0][1].item() + result[0][0].item()
fall_out = fp / fp_tn
print(f"fall Out: {fall_out}")
```

# Precision  精确率
精确率的公式:
$$
Precision = \frac{被预测分类的实际正样本}{所有被预测为正样本的样本} = \frac{TP}{TP+FP}
$$
```python
tp_fp = 0
tp_fp += result[1][1].item() + result[0][1].item()
precision = tp / tp_fp
```

```python
# 安装seaborn库
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=Y)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

import numpy as np

classes, counts = np.unique(Y, return_counts=True)

for cls, count in zip(classes, counts):
    print(f"Class {cls}: {count} ({count / len(Y):.2%})")
```