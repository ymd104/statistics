import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson
import matplotlib_fontja

matplotlib_fontja.japanize()

np.random.seed(0)
n = 200
beta_0 = 1.0
beta_1 = 2.0

x = np.linspace(0, 10, n)
X = sm.add_constant(x)


def generate_errors_ar1(rho, n):
    u = np.random.normal(0, 1, n)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = rho * e[t-1] + u[t]
    return e


# ホワイトノイズ
eps_iid = np.random.normal(0, 1, n)
y_iid = beta_0 + beta_1 * x + eps_iid
model_iid = sm.OLS(y_iid, X).fit()

# 自己相関あり（AR(1)）
rho = -0.7
eps_ar = generate_errors_ar1(rho, n)
y_ar = beta_0 + beta_1 * x + eps_ar
model_ar = sm.OLS(y_ar, X).fit()


print("=== IID（誤差独立）===")
print(model_iid.summary().tables[1])

print(f"\n=== AR(1) 誤差（ρ = {rho}）===")
print(model_ar.summary().tables[1])


t = np.arange(n)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t, eps_iid, color='skyblue')
plt.title('ホワイトノイズの誤差（iid）')
plt.xlabel('時刻')
plt.ylabel('誤差')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t, eps_ar, color='salmon')
plt.title(f'自己相関ありの誤差（AR(1), ρ={rho}）')
plt.xlabel('時刻')
plt.ylabel('誤差')
plt.grid(True)

plt.tight_layout()
plt.show()


se_iid_list = []
se_ar_list = []

for _ in range(1000):
    eps_iid = np.random.normal(0, 1, n)
    y_iid = beta_0 + beta_1 * x + eps_iid
    model_iid = sm.OLS(y_iid, X).fit()
    se_iid_list.append(model_iid.bse[1])

    eps_ar = generate_errors_ar1(rho, n)
    y_ar = beta_0 + beta_1 * x + eps_ar
    model_ar = sm.OLS(y_ar, X).fit()
    se_ar_list.append(model_ar.bse[1])

print(f"平均標準誤差（iid）: {np.mean(se_iid_list):.4f}")
print(f"平均標準誤差（AR(1)）: {np.mean(se_ar_list):.4f}")

# 残差の取得
resid_iid = model_iid.resid
resid_ar = model_ar.resid

# DW比の計算
dw_iid = durbin_watson(resid_iid)
dw_ar = durbin_watson(resid_ar)

# 結果表示
print(f"DW比（iid 誤差）: {dw_iid:.4f}")
print(f"DW比（AR(1) 誤差）: {dw_ar:.4f}")
