"""Google 机器学习公平性练习。"""

# 在这里跟随课程逐步添加练习代码。
# 如果练习需要本地数据集，请统一放在仓库根目录的 data 文件夹中。
import pandas as pd
import tensorflow as tf

import tensorflow_model_analysis as tfma
from google.protobuf import text_format

from tensorflow_model_remediation import min_diff
