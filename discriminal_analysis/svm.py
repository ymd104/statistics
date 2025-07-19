import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

meanA = [0, 0]
covA = [[1, 0], [0, 1]]

meanB = [2, 2]
covB = [[0.5, 0], [0, 0.5]]

rvA = multivariate_normal(mean=meanA, cov=covA)
rvB = multivariate_normal(mean=meanB, cov=covB)

np.random.seed(0)
dataA = rvA.rvs(size=150)
dataB = rvB.rvs(size=20)

X = np.vstack((dataA, dataB))
y_true = np.hstack((np.zeros(150), np.ones(20)))

svm = SVC(kernel="rbf", C=1.0, gamma="scale")
svm.fit(X, y_true)

y_pred = svm.predict(X)

cm = confusion_matrix(y_true, y_pred)
print(cm)

misclassification_rate = 1 - np.trace(cm) / np.sum(cm)
print(f"{misclassification_rate:.4f}")

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = svm.predict(grid).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2, levels=[-1, 0, 1], colors=["blue", "orange"])
plt.scatter(dataA[:, 0], dataA[:, 1], alpha=0.6, label="Group A")
plt.scatter(dataB[:, 0], dataB[:, 1], alpha=0.6, label="Group B")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("SVM")
plt.grid(True)
plt.axis("equal")
plt.show()
