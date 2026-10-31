import pandas as pd
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
auto_mpg = fetch_ucirepo(id=9) 
  
# data (as pandas dataframes) 
X:pd.DataFrame = auto_mpg.data.features 
Y:pd.DataFrame  = auto_mpg.data.targets 

# # metadata 
# print(auto_mpg.metadata) 
  
# # variable information 
# print(auto_mpg.variables) 

# print(y.columns)
# displacement	发动机排量
# cylinders	气缸数量
# horsepower	马力
# weight	汽车重量
# acceleration	加速性能
# model_year	车型年份
# origin	产地

# mpg，燃油效率。

y = Y["mpg"][0] / 1000
x = X["weight"][0]
b = 46.3021
w = -7.6612

y_pre = b + (w * x)

print(f"y_pre = {y_pre}\n", f"y = {y}")

l2_loss = (x - y)**2

print(f"l2_loss = {l2_loss}")
