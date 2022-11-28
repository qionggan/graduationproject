import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import Normalizer

matplotlib.rcParams['font.family'] = 'SimHei'

import scipy.cluster.hierarchy as shc

# 1.数据加载与处理
data = pd.read_csv('Wholesale customers data.csv', encoding='utf-8')
data.head(10)
data.describe()
data.shape
# 检查唯一值
data.nunique()
# 检查缺失值，如果有缺失值，我们用fillna方法来填充数据集中的空值
data.isnull().sum()
# 检查数据类型
data.info()
# 数据的归一化
scaler = Normalizer().fit(data)
data_scaled = scaler.transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

# 2.谱线图绘制
plt.figure(figsize=(10, 7))
plt.title("Dendrograms")  # 树状图
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.axhline(y=6, color='r', linestyle='--') # 添加横线

# 3.模型训练：层次聚类
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  # 实例化层次聚类模型构建
cluster_pre = cluster.fit_predict(data_scaled)  # 模型训练和预测
print(cluster_pre)
cluster_count = pd.value_counts(cluster_pre)  # 不同簇的样本个数
print(cluster_count)
# plt.figure(figsize=(5, 5))
# cluster_count.plot.bar()

# 4.聚类结果可视化
plt.figure(figsize=(10, 7))
plt.scatter(data_scaled['Milk'], data_scaled['Grocery'], c=cluster.labels_)
plt.xlabel("Milk")
plt.ylabel("Grocery")
plt.show()
plt.close()

# 5.模型评估：轮廓系数SC
from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(data_scaled, cluster.labels_)  # 轮廓系数均值
print('silhouette score: {:<.3f}'.format(silhouette_avg))

# 绘制轮廓系数图查看聚类效果
sample_silhouette_values = silhouette_samples(data_scaled, cluster.labels_)

# graph_component_silhouette(2, [-0.15, 0.55], len(data_scaled), sample_silhouette_values,cluster.labels_)
X = data_scaled
n_clusters = 4
y_lower = 10
for i in range(n_clusters):
    # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values[cluster.labels_ == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / n_clusters)
    plt.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )
    # Label the silhouette plots with their cluster numbers at the middle
    # plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

plt.title("The silhouette plot for the various clusters",fontsize=28)
plt.xlabel("The silhouette coefficient values",fontsize=24)
plt.ylabel("Cluster label",fontsize=24)

# The vertical line for average silhouette score of all the values
plt.axvline(x=silhouette_avg, color="red", linestyle="--")
plt.yticks([])  # Clear the yaxis labels / ticks
plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.show()

# 寻找最优簇
AgglomerativeClustering_per_c = [AgglomerativeClustering(n_clusters=c).fit(data_scaled) for c in range(1, 10)]
silhouette_scores = [silhouette_score(data_scaled, model.labels_) for model in AgglomerativeClustering_per_c[1:]]
plt.figure(figsize=(8, 4))
plt.plot(range(2, 10), silhouette_scores, 'bo-')
plt.show()

# 6.雷达图的绘制来可视化聚类的特点
data_scaled_new = pd.DataFrame(data, columns=data.columns)
data_scaled_new['label'] = cluster.labels_
data_scaled_new = data_scaled_new.groupby('label').mean().applymap(lambda x: '%.2f' % x)
data_scaled_new.shape

# 数据框行列转置
data_T = pd.DataFrame(data_scaled_new.values).T
data_df = pd.DataFrame(data_T.values, index=data_scaled_new.columns, columns=['0', '1'])
radar_labels = data_df.index
nAttr = 8
data = data_df.values.astype(float)  # 数据类型转换
data_labels = data_df.columns
# 设置角度
angles = np.linspace(0, 2 * np.pi, nAttr, endpoint=False)
data = np.concatenate((data, [data[0]]))
data.shape
angles = np.concatenate((angles, [angles[0]]))
angles.shape
# 设置画布
fig = plt.figure(facecolor="white", figsize=(10, 6))
plt.subplot(111, polar=True)
# 绘图
plt.plot(angles, data, 'o-', linewidth=1.5, alpha=0.2)
# 填充颜色
plt.fill(angles, data, alpha=0.25)
plt.thetagrids(angles[:-1] * 180 / np.pi, radar_labels, 1.2)
plt.figtext(0.52, 0.95, '客户不同的聚类的特点', ha='center', size=20)
# 设置图例
legend = plt.legend(data_labels, loc=(1.1, 0.05), labelspacing=0.1)
plt.setp(legend.get_texts(), fontsize='large')
plt.grid(True)
plt.savefig('聚类雷达图.png')
plt.show()
