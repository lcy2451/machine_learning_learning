import matplotlib.pyplot as plt

import torch
import pandas as pd
from ucimlrepo import fetch_ucirepo 

if __name__ == "__main__":
    # fetch dataset 
    seoul_bike_sharing_demand = fetch_ucirepo(id=560) 
    
    # data (as pandas dataframes) 
    X:pd.DataFrame = seoul_bike_sharing_demand.data.features 
    y:pd.DataFrame = seoul_bike_sharing_demand.data.targets 

    z_score_scaling = (X["Temperature"] - X["Temperature"].mean()) / X["Temperature"].std(ddof=0)
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(X["Temperature"], bins=20)

    axes[1].hist(z_score_scaling, bins=20)
    plt.show()
