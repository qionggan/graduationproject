import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# 设置好参数
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

# 导入数据集合
iris = load_iris()
plt.figure(figsize=(15,10))
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    # 选取二维数据进行两两比较
    X = iris.data[:, pair]
    y = iris.target

    # 建立并训练模型
    clf = DecisionTreeClassifier().fit(X, y)

    # 画出决策边界
    print(pairidx)#0,1,2,3,4,5,
    plt.subplot(2, 3, pairidx + 1)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel(iris.feature_names[pair[0]], fontsize= 20)
    plt.ylabel(iris.feature_names[pair[1]], fontsize= 20)

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=100)

plt.suptitle("Decision surface of a decision tree using paired features", y=1.02, fontsize= 25)
plt.legend(loc='lower right', borderpad=0, handletextpad=0, fontsize= 15)
plt.axis("tight")