import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib_fontja

matplotlib_fontja.japanize()

X, y, coef_true = make_regression(
    n_samples=100, n_features=50, noise=5.0, coef=True, random_state=0
)
X = StandardScaler().fit_transform(X)

alphas = np.logspace(-3, 3, 100)

ridge_nonzero = []
lasso_nonzero = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha).fit(X, y)
    lasso = Lasso(alpha=alpha, max_iter=10000).fit(X, y)

    ridge_nonzero.append(np.sum(np.abs(ridge.coef_) > 1e-4))
    lasso_nonzero.append(np.sum(np.abs(lasso.coef_) > 1e-4))

plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_nonzero, label="Ridge: #non-zero coef", color="blue")
plt.plot(alphas, lasso_nonzero, label="Lasso: #non-zero coef", color="red")
plt.xscale("log")
plt.xlabel("alpha (正則化の強さ)")
plt.ylabel("非ゼロ係数の数")
plt.title("正則化パラメータに対する非ゼロ係数数の変化")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
