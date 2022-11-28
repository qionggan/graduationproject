from sklearn.datasets import load_iris
import numpy as np
iris = load_iris()
data = iris.data
n = len(data)
k = 3
dist = np.zeros([n, k+1])
# 1、选中心
center = data[:k, :]
center_new = np.zeros([k, data.shape[1]])
while True:
    # 2、求距离
    for i in range(n):
        for j in range(k):
            dist[i, j] = np.sqrt(sum((data[i, :] - center[j, :])**2))
        dist[i, k] = np.argmin(dist[i, :k])   # 3、归类

    for i in range(k):   # 4、求新类中心
        index = dist[:, k] == i
        center_new[i, :] = data[index, :].mean(axis=0)
    if np.all(center == center_new):   # 5、判定结束
        break
    center = center_new
print(dist[:,k])



