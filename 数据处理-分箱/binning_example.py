import matplotlib.pyplot as plt

import pandas as pd

from ucimlrepo import fetch_ucirepo 

if __name__ == "__main__":
    print("xx")
    # fetch dataset 
    iris = fetch_ucirepo(id=53) 
    
    # data (as pandas dataframes) 
    X:pd.DataFrame = iris.data.features 
    Y:pd.DataFrame = iris.data.targets 

    print(type(X["petal length"]))
    # 按顺序查找可用字体，兼容 macOS 和 Windows
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",  # Windows：微软雅黑
        "PingFang SC",      # macOS：苹方
        "Heiti SC",         # macOS：黑体
        "SimHei",          # Windows：黑体
        "DejaVu Sans",
    ]
    plt.xlabel("petal length 花瓣长度")
    plt.ylabel("petal width 花瓣宽度")
    plt.scatter(
        X["petal length"],
        X["petal width"]
        )
    plt.axvline(
        x=2.5,
        color="red",
        linestyle="--",
        alpha=0.7
    )
    plt.show()
