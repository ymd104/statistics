import numpy as np
import matplotlib.pyplot as plt

import matplotlib_fontja

matplotlib_fontja.japanize()


mu = np.array([0, 0])
sigma_x = 1.0
sigma_y = 1.0
rho = 0.1

cov = np.array([
    [sigma_x**2, rho * sigma_x * sigma_y],
    [rho * sigma_x * sigma_y, sigma_y**2]
])


np.random.seed(1)
samples = np.random.multivariate_normal(mu, cov, size=2000)

x_vals = samples[:, 0]
y_vals = samples[:, 1]

x_line = np.linspace(-3, 3, 100)
E_y_given_x = mu[1] + rho * (sigma_y / sigma_x) * (x_line - mu[0])


plt.figure(figsize=(8, 6))
plt.scatter(x_vals, y_vals, alpha=0.3, label="2変量正規分布からのサンプル")
plt.plot(x_line, E_y_given_x, color="red", label=r"$E[Y|X]$（条件付き期待値）", linewidth=2)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("2変量正規分布と条件付き期待値 $E[Y|X]$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
