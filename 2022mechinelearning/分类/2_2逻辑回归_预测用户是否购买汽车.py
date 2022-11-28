import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 1.导入数据集
dataset = pd.read_csv('Social_Network_Ads.csv')
# dataset.head()
# dataset.describe()
X = dataset.iloc[:, [2,3]].values
Y = dataset.iloc[:,4].values
# 将数据集分成训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
# 特征缩放
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# 2.逻辑回归模型
# 将逻辑回归应用于训练集
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
# 3.预测
# 测试集结果
y_pred = classifier.predict(X_test)
# 4.模型的评估
# 生成混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# # 5.决策边界和样本点可视化
# from matplotlib.colors import ListedColormap
# X_set,y_set=X_train,y_train
# X1,X2=np. meshgrid(np. arange(start=X_set[:,0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
#                    np. arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1, step=0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(),X1.max())
# plt.ylim(X2.min(),X2.max())
# for i,j in enumerate(np. unique(y_set)):
#     plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1], c = ListedColormap(('red', 'yellow'))(i), label=j)
#
# plt. title(' LOGISTIC(Training set)')
# plt. xlabel(' Age')
# plt. ylabel(' Estimated Salary')
# plt. legend()
# plt. show()
# plt.close()


# X_set,y_set=X_test,y_test
# X1,X2=np. meshgrid(np. arange(start=X_set[:,0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
#                    np. arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1, step=0.01))
#
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(),X1.max())
# plt.ylim(X2.min(),X2.max())
# for i,j in enumerate(np. unique(y_set)):
#     plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c = ListedColormap(('red', 'green'))(i), label=j)
#
# plt. title(' LOGISTIC(Test set)')
# plt. xlabel(' Age')
# plt. ylabel(' Estimated Salary')
# plt. legend()
# plt. show()

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
plt.subplot(1,2,2)
plt.plot(fpr, tpr, marker='o')
plt.xlabel('False Positive Rate', size=18)
plt.ylabel('True Positive Rate', size=18)
plt.title('ROC Curve', size=18)
