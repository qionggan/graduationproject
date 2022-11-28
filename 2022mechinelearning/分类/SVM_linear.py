from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['axes.unicode_minus']=False

#定义决策超平面函数可视化函数：
def plot_svc_decision_function(model,ax=None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = np.linspace(xlim[0],xlim[1],30)#1.画图x轴取值范围
    y = np.linspace(ylim[0],ylim[1],30)#1.画图y轴取值范围
    Y,X = np.meshgrid(y,x)#2.构成二维网格点 坐标矩阵 画棋盘（30*30）
    print(Y.shape)
    # xy = np.vstack([X.ravel(), Y.ravel()]).T
    xy = np.c_[X.ravel(), Y.ravel()]  #3.拉平并按列连接数组（900*2）
    print(xy.shape)
    P = model.decision_function(xy).reshape(X.shape)#4.决策函数（30*30）到分离超平面的有符号距离
    print(P.shape)
    # P = model.predict(xy).reshape(X.shape)  # 预测结果
    # plt.contour(X, Y, P, [0, 0.01])
    ax.contour(X, Y, P,colors="k",levels=[-1,0,1],alpha=0.8,linestyles=["--","-","--"])#5.等高线
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

#训练数据集创建
X,y = make_blobs(n_samples=50, centers=2, random_state=0,cluster_std=0.6)#构建训练数据
clf = SVC(kernel = "linear").fit(X,y)#SVM线性核模型的构建和训练
supports = clf.support_vectors_#SVC可直接得到支持向量
# clf.support_ndarray
supports
plt.figure(figsize=(10,6))
plt.scatter(X[:,0],X[:,1],c=y,s=80,cmap="copper")#样本点
plot_svc_decision_function(clf)
plt.scatter(supports[:,0],supports[:,1],marker='o',s=200,facecolors='y', edgecolors='r')#支持向量