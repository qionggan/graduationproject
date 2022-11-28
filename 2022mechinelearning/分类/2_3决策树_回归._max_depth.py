import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
# 创建数据集 Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(100, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(20)) #添加噪声

# # 回归决策树模型的训练
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=10)
regr_1.fit(X, y)
regr_2.fit(X, y)

# 预测
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
X_test.shape
y_test = np.sin(X_test).ravel()
y_test[::5] += 3 * (0.5 - rng.rand(100))
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# 结果可视化
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=10", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()


import graphviz
from sklearn import tree
plt.subplot(1,2,1)
tree.plot_tree(regr_1,filled=True)#决策树可视化
print(regr_1.score(X_test,y_test))#R方系数


plt.subplot(1,2,2)
tree.plot_tree(regr_2,filled=True)#决策树可视化
# v=((y_test - y_test.mean()) ** 2).sum()
# u=((y_test- y_2) ** 2).sum()
# r2_2=1-u/v #R方系数定义计算
print(regr_2.score(X_test,y_test))#R方系数


#max-depth 学习曲线
train_list = []
test_list = []
for i in range(10):
    clf = DecisionTreeRegressor(random_state=25
                                 ,max_depth=i+1
                                )
    clf = clf.fit(X, y)
    score_train = clf.score(X,y)
    score_test = clf.score(X_test, y_test)
    train_list.append(score_train)
    test_list.append(score_test)
print(max(test_list))
plt.figure(figsize=(10, 6))
plt.plot(range(1,11),train_list,color="orange",label="train")
plt.plot(range(1,11),test_list,color="black",label="test")
plt.xticks(range(1,11))
plt.legend()
plt.show()


