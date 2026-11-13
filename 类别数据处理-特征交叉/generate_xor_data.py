
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 参数
num_samples = 400
noise = 0.0
seed = 42

rng = np.random.default_rng(seed)

# 生成坐标，对应 Playground 的 [-5, 5]
x1 = rng.uniform(-5, 5, num_samples)
x2 = rng.uniform(-5, 5, num_samples)

# 坐标轴附近增加 0.3 的间隔
padding = 0.3
x1 += np.where(x1 > 0, padding, -padding)
x2 += np.where(x2 > 0, padding, -padding)

# 噪声仅影响标签判定，与 Playground 一致
noise_x = rng.uniform(-5, 5, num_samples) * noise
noise_y = rng.uniform(-5, 5, num_samples) * noise

# XOR 分类：同号为 1，异号为 -1
labels = np.where(
    (x1 + noise_x) * (x2 + noise_y) >= 0,
    1,
    -1
)

# 导出 CSV
df = pd.DataFrame({
    "x1": x1,
    "x2": x2,
    "label": labels
})

df.to_csv(Path(__file__).parent.parent / "data"/ "playground_xor.csv", index=False)
print(df.head())

# 可视化
plt.figure(figsize=(7, 7))

plt.scatter(
    df.loc[df["label"] == 1, "x1"],
    df.loc[df["label"] == 1, "x2"],
    label="Class +1",
    alpha=0.8
)

plt.scatter(
    df.loc[df["label"] == -1, "x1"],
    df.loc[df["label"] == -1, "x2"],
    label="Class -1",
    alpha=0.8
)

plt.axhline(0, linewidth=0.8, color="gray")
plt.axvline(0, linewidth=0.8, color="gray")

plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("TensorFlow Playground - XOR Dataset")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
