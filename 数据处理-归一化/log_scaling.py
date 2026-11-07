import matplotlib.pyplot as plt

import torch
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 

if __name__ == "__main__":
    # fetch dataset 
    online_news_popularity = fetch_ucirepo(id=332)  
    
    # data (as pandas dataframes) 
    X:pd.DataFrame = online_news_popularity.data.features 
    y:pd.DataFrame = online_news_popularity.data.targets 
    y.columns = y.columns.str.strip()

    shares = y["shares"].sample(n=5000, random_state=42)
    log_scaling = np.log(shares)

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(shares, bins=50)

    axes[1].hist(log_scaling,bins=50)
    plt.show()
