from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression # 线性回归模型
from sklearn.model_selection import train_test_split# 交叉验证
X = load_boston().data
y = load_boston().target
# df=pd.DataFrame(X,columns=load_boston().feature_names)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3,random_state=420)
#先查看方差的变化，再看R2的变化
alpha_range = np.arange(1,1001,100)
ridge_var, lr_var, ridge_mean, lr_mean = [], [], [], []
plt.figure(figsize=(16,10))
plt.style.use('seaborn')
for alpha in alpha_range:
    ridge = Ridge(alpha=alpha)
    linear = LinearRegression()
    score_rig = cross_val_score(ridge,X,y,cv=5,scoring="r2")
    score_linear = cross_val_score(linear,X,y,cv=5,scoring="r2")
    ridge_var.append(score_rig.var())
    lr_var.append(score_linear.var())
    ridge_mean.append(score_rig.mean())
    lr_mean.append(score_linear.mean())
name = ['variance', 'mean']
for i,j in enumerate([[ridge_var, lr_var], [ridge_mean, lr_mean]]):
    plt.subplot(1,2,i+1)
    plt.plot(alpha_range,j[0],color="green",label="Ridge")
    plt.plot(alpha_range,j[1],color="blue",label="LR")
    plt.title(f"cross_val_score {name[i]} for different alpha values")
    plt.xlabel("alpha")
    plt.ylabel(f"{name[i]}")
    plt.legend()
plt.show()
# ridge = Ridge(alpha=175)
# ridge.fit(Xtrain,Ytrain)
# pre=ridge.predict(Xtest)
# pre
# #细化学习曲线
# alpha_range = np.arange(100,300,10)
# ridge_score, lr_score = [], []
# for alpha in alpha_range:
#     reg = Ridge(alpha=alpha)
#     linear = LinearRegression()
#     regs = cross_val_score(reg,X,y,cv=5,scoring = "r2").mean()
#     linears = cross_val_score(linear,X,y,cv=5,scoring = "r2").mean()
#     ridge_score.append(regs)
#     lr_score.append(linears)
# for i,j in enumerate([[ridge_score,"Ridge", "green"], [lr_score, "LR", 'blue']]):
#     plt.subplot(2,2,i+3)
#     plt.plot(alpha_range,j[0],color=j[2],label=j[1])
#     plt.title("cross_val_score mean for different alpha values")
#     plt.xlabel("alpha")
#     plt.ylabel("Mean")
#     plt.legend()
# plt.suptitle("cross_val_score of the LR and Ridge for different alpha values", y=0.95, fontsize=18)
# plt.show()

# from sklearn import datasets, linear_model
# from sklearn.model_selection import cross_val_score
# diabetes = datasets.load_diabetes()
# X = diabetes.data[:150]
# y = diabetes.target[:150]
# lasso = linear_model.Lasso()
# print(cross_val_score(lasso, X, y, cv=None))
