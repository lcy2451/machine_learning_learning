from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    _csv = Path(__file__).parent.parent / "data"/ "playground_xor.csv"

    _df = pd.read_csv(filepath_or_buffer=_csv)

    x1 = _df["x1"]
    x2 = _df["x2"]

    x3 = _df["x3"] = x1 * x2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(
        _df.loc[_df["label"] == 1, "x1"],
        _df.loc[_df["label"] == 1, "x2"],
        label="Class +1",
        alpha=0.8
    )

    axes[0].scatter(
        _df.loc[_df["label"] == -1, "x1"],
        _df.loc[_df["label"] == -1, "x2"],
        label="Class -1",
        alpha=0.8
        )

    axes[1].scatter(
        _df.loc[_df["label"] == 1, "x1"],
        _df.loc[_df["label"] == 1, "x3"],
        label="Class +1",
        alpha=0.8
        )
    
    axes[1].scatter(
        _df.loc[_df["label"] == -1, "x1"],
        _df.loc[_df["label"] == -1, "x3"],
        label="Class +1",
        alpha=0.8
        )

    axes[0].set_title("Original Features")
    axes[1].set_title("Feature Cross")

    axes[1].axhline(y=0, color="black", linestyle="--")
    
    plt.show()

