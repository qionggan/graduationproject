# 随机森林-特征重要性
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from scipy import io as spio
import pandas as pd
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
import numpy as np

#===============================================================
# 1.鸢尾花分类特征重要性分析
iris = load_iris()
rf_clf = RandomForestClassifier(n_estimators=500,n_jobs=-1)
rf_clf.fit(iris['data'],iris['target'])
for name,score in zip(iris['feature_names'],rf_clf.feature_importances_):#zip函数压缩连个列表，实现并行遍历
    print (name,score)
    plt.barh(y=name,width=score,height=0.4)
    # plt.bar(x=name,height=score,orientation="horizontal")

# 特征重要性可视化
feature_imp = pd.Series(
    rf_clf.feature_importances_,
    index=iris['feature_names']).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12,8))
sns.barplot(x=feature_imp,y=feature_imp.index)#条形图
plt.xlabel('Feature Importance Score',fontsize=20)
plt.ylabel('Features')
plt.legend(iris['feature_names'],fontsize=15)
plt.show()


#========================================================================
# 2.手写数字图片重要区域识别分析（特征重要性展示）
mnist = spio.loadmat('mnist-original.mat') #加载手写数字图片
mnist['data'].shape
mnist['label'].shape
image_0=mnist['data'][:,42].reshape(28,28)#手写数字data 进行reshape
plt.imshow(image_0)            #手写数字图片可视化

rf_clf = RandomForestClassifier(n_estimators=100,n_jobs=-1)#模型构建
rf_clf.fit(mnist['data'].T,mnist['label'].ravel())#模型训练，训练数据矩阵格式转换
rf_clf.feature_importances_.shape
def plot_digit(data):
    image = data.reshape(28,28)
    plt.imshow(image,cmap=matplotlib.cm.hot)
    plt.axis('off')
plot_digit(rf_clf.feature_importances_)
char = plt.colorbar(ticks=[rf_clf.feature_importances_.min(), rf_clf.feature_importances_.max()])
char.ax.set_yticklabels(['Not important', 'Very important'])
plt.show()