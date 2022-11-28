import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from SVM_linear import plot_svc_decision_function
# 导入数据
iris=datasets.load_iris()
iris
X = iris["data"][:,(2,3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris-Viginica true or false 判断并类型转化为float64
y
y.shape
# 模型训练
svm_clf = Pipeline((
    ('std',StandardScaler()),
    ('linear_svc',LinearSVC(C=1))
))
svm_clf.fit(X,y)

# 对比不同C值所带来的效果差异
scaler = StandardScaler()
svm_clf1 = LinearSVC(C=1,random_state = 42)
svm_clf2 = LinearSVC(C=100,random_state = 42)

scaled_svm_clf1 = Pipeline((
    ('std',scaler),
    ('linear_svc',svm_clf1)
))

scaled_svm_clf2 = Pipeline((
    ('std',scaler),
    ('linear_svc',svm_clf2)
))
scaled_svm_clf1.fit(X,y)
scaled_svm_clf2.fit(X,y)

# b截距和w参数(LinearSVC对象是没有.support_vectors_属性（即支持向量点），所以我们需要自己来定义)
## Convert to unscaled parameters
b1 = svm_clf1.decision_function([-scaler.mean_ / scaler.scale_])#w1*x1+w2*x2+b=1和w1*x1+w2*x2+b=-1相加得b
b2 = svm_clf2.decision_function([-scaler.mean_ / scaler.scale_])
w1 = svm_clf1.coef_[0] / scaler.scale_
w1.shape
w2 = svm_clf2.coef_[0] / scaler.scale_
svm_clf1.intercept_ = np.array([b1])
svm_clf2.intercept_ = np.array([b2])
svm_clf1.coef_ = np.array([w1])
svm_clf1.coef_.shape
svm_clf2.coef_ = np.array([w2])

# 支持向量(LinearSVC does not do this automatically)
# y的取值是[0,1]，y_的取值变为[-1,1]，符合SVM的一般形式
y_1 = y * 2 - 1
# 在这里设定在边界内的点都是support_vector
support_vectors_idx1 = (y_1 * (X.dot(w1) + b1) < 1).ravel()#支持向量的索引
support_vectors_idx2 = (y_1 * (X.dot(w2) + b2) < 1).ravel()
svm_clf1.support_vectors_ = X[support_vectors_idx1]
svm_clf2.support_vectors_ = X[support_vectors_idx2]
svm_clf2.support_vectors_.shape
svm_clf1.support_vectors_.shape
# 决策超平面可视化
plt.figure(figsize=(14,4.2))
plt.subplot(121)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", label="Iris-Virginica")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs", label="Iris-Versicolor-Setosa")
plot_svc_decision_function(svm_clf1,ax=None)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.title("$C = {}$".format(svm_clf1.C), fontsize=16)#C字符串格式化输出
plt.axis([4, 6, 0.8, 2.8])

plt.subplot(122)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
plot_svc_decision_function(svm_clf2, ax=None)
plt.xlabel("Petal length", fontsize=14)
plt.title("$C = {}$".format(svm_clf2.C), fontsize=16)
plt.axis([4, 6, 0.8, 2.8])

