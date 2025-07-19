import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib_fontja

matplotlib_fontja.japanize()


# 関数定義
def f(x):
    return norm.pdf(x)


def g(x):
    return x * (1 - norm.cdf(x))


# x軸の範囲（g(x)はxが正のときが意味ありそう）
x = np.linspace(-4, 4, 500)

# yの値の計算
y_f = f(x)
y_g = g(x)

# グラフ描画
plt.figure(figsize=(8, 5))
plt.plot(x, y_f, label=r"$f(x) = \varphi(x)$", color="blue")
plt.plot(
    x, y_g, label=r"$g(x) = x \cdot \left[(1 - \Phi(x)) - x\right]$", color="orange"
)

# 装飾
plt.title("f(x) と g(x) のグラフ")
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(0, color="gray", linestyle="--")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
