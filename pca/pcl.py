import pandas as pd
import numpy as np

# データの作成
data = {
    "x_1": [2, 9, 8, 7, 2, 5],  # 国語
    "x_2": [2, 8, 3, 1, 9, 4],  # 数学
    "x_3": [3, 10, 2, 3, 8, 5],  # 理科
    "x_4": [1, 9, 7, 8, 2, 5],  # 社会
}

df = pd.DataFrame(data, index=[f"person_{i+1}" for i in range(6)])

# 分散共分散行列
cov_matrix = df.cov()

# 相関行列
corr_matrix = df.corr()

print("【分散共分散行列】")
print(cov_matrix)

print("\n【相関行列】")
print(corr_matrix)


eigvals, eigvecs = np.linalg.eig(corr_matrix)

# 固有値と固有ベクトルが正しいかの検証
print("\n固有値:")
print(eigvals)

print("\n固有ベクトル（列ベクトルが主成分ベクトル）:")
for i in range(len(eigvals)):
    print(f"主成分 {i+1}: {eigvecs[:, i]}")

for i in range(len(eigvals)):
    left = corr_matrix @ eigvecs[:, i]
    right = eigvals[i] * eigvecs[:, i]
    print(f"\n主成分 {i+1}:")
    print("Ru:")
    print(left)
    print("λu:")
    print(right)


# 主成分負荷量(定義による算出)
loadings = eigvecs * np.sqrt(eigvals)

loadings_df = pd.DataFrame(
    loadings, columns=[f"PC{i+1}" for i in range(loadings.shape[1])], index=df.columns
)
print("\n主成分負荷量（factor loadings）:")
print(loadings_df)


# 主成分スコア（各主成分における各サンプルの値）を計算
X = pd.DataFrame(data)
Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
scores = Z @ eigvecs

# 各変数と各主成分との相関係数を計算
loadings_from_corr = np.corrcoef(Z.T, scores.T)[0 : Z.shape[1], Z.shape[1] :]

print("相関係数から求めた主成分負荷量（主成分と元の変数の相関）:")
print(loadings_from_corr)
