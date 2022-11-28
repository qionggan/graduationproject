import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# 生产数据函数
#
# def uniform(size):
#     x = np.linspace(0,1,size)
#     return x.reshape(size,1)
#
# def create_data(size):
#     x = uniform(size)
#     np.random.seed(42) #设置随机数种子
#     y = sin_fun(x)+np.random.normal(scale=0.25, size=x.shape)
#     return x,y
#
# def sin_fun(x):
#     return np.sin(2*np.pi*x)
#
# X,y = create_data(200) # 利用上面的生产数据函数

def myfun(x):
    '''目标函数
    input:x(float):自变量
    output:函数值'''
    return 10 + 5 * x + 4 * x**2 + 6 * x**3
X = np.linspace(-3,3, 50)

np.random.seed(42) #设置随机数种子
y = myfun(X) + np.random.random(size=len(X)) * 100 - 50#目标函数上加上噪声产生样本点
# y = myfun(X)+np.random.normal(scale=0.25, size=X.shape)
X= X.reshape(-1,1)
y = y.reshape(-1,1)
# fig = plt.figure(figsize=(10,6))
for i,degree in enumerate([1,3,5,9]):
    # 利用Pipeline将三个模型封装起来串联操作
    poly_reg = Pipeline([
                        ("poly", PolynomialFeatures(degree=degree)),
                        ("std_scaler", StandardScaler()),
                        ("lin_reg", LinearRegression())
                        ])
    poly_reg.fit(X, y)            #训练
    y_pred = poly_reg.predict(X)  #预测
    plt.subplot(2, 2, i+1)
    plt.scatter(X,y,facecolor="none", edgecolor="g", s=35, label="training data")
    plt.plot(X,y_pred,c="orange",label="fitting")
    plt.title("degree={}".format(degree))
    plt.legend(loc="best")
    plt.ylabel("Regression output",fontsize=13)
    plt.xlabel("Input feature",fontsize=13)
    plt.pause(1)
plt.show()


