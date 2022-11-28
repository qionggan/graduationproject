import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cluster import dbscan

#1.产生数据集及可视化
X,_ = datasets.make_moons(500,noise = 0.1,random_state=1)
df = pd.DataFrame(X,columns = ['feature1','feature2'])
df.plot.scatter('feature1','feature2', s = 100,alpha = 0.6, title = 'dataset by make_moon')

#2.调用dbscan函数进行模型训练
# eps为邻域半径，min_samples为最少点数目
core_samples,cluster_ids = dbscan(X, eps = 0.2, min_samples=20)
# cluster_ids中-1表示对应的点为噪声点

print(core_samples)
print(cluster_ids)
df = pd.DataFrame(np.c_[X,cluster_ids],columns = ['feature1','feature2','cluster_id'])
df['cluster_id'] = df['cluster_id'].astype('i2')

df.plot.scatter('feature1','feature2', s = 100,
    c = list(df['cluster_id']),cmap = 'rainbow',colorbar = False,
    alpha = 0.6,title = 'DBSCAN cluster result')