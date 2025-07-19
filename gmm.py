import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import matplotlib_fontja

matplotlib_fontja.japanize()

# 正規分布のパラメータ
mu1, sigma1 = -2, 1
mu2, sigma2 = 2, 1

# x軸の範囲
x = np.linspace(
    min(mu1 - 3 * sigma1, mu2 - 3 * sigma2) - 1,
    max(mu1 + 3 * sigma1, mu2 + 3 * sigma2) + 1,
    1000,
)

# それぞれの分布の確率密度
pdf1 = norm.pdf(x, mu1, sigma1)
pdf2 = norm.pdf(x, mu2, sigma2)

# 線の描画（←ここを完成させてください）
sum_pdf = 0.5 * pdf1 + 0.5 * pdf2

# プロット
plt.plot(
    x,
    pdf1,
    label=f"N({mu1},{sigma1}^2)",
    linestyle="--",
)
plt.plot(
    x,
    pdf2,
    label=f"N({mu2},{sigma2}^2)",
    linestyle="--",
)
plt.plot(x, sum_pdf, label="Sum of distributions", color="black")
plt.legend()
plt.title("1次元混合ガウス分布")
plt.grid()
plt.show()
