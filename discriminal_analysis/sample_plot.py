import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import matplotlib_fontja

matplotlib_fontja.japanize()

mean1 = [0, 0]
cov1 = [[1, 0], [0, 1]]

mean2 = [2, 2]
cov2 = [[0.5, 0], [0, 0.5]]

rv1 = multivariate_normal(mean=mean1, cov=cov1)
rv2 = multivariate_normal(mean=mean2, cov=cov2)

np.random.seed(0)
data1 = rv1.rvs(size=150)
data2 = rv2.rvs(size=20)

plt.scatter(data1[:, 0], data1[:, 1], alpha=0.6, label="Group A")
plt.scatter(data2[:, 0], data2[:, 1], alpha=0.6, label="Group B")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("2元正規分布のサンプル")
plt.grid(True)
plt.axis("equal")
plt.show()
