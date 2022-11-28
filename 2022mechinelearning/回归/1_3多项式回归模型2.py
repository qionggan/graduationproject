import matplotlib.pyplot as  plt
import numpy as np

# 先拟定一个一元三次多项式作为目标函数
def myfun(x):
    """

    :param x: 自变量
    :return: 函数值
    """
    return 10+5*x+4*x**2+6*x**3
# 然后再加上一些噪声产生样本集
x_train=np.linspace(-3,3,15)
y_train=myfun(x_train)+np.random.random(size=len(x_train))*100-50#产生噪声样本集
# plt.subplots(2,2,1)
plt.scatter(x_train,y_train,color='green',linewidths=2,label='样本点')#样本点的可视化
plt.legend()
plt.rc('font',family=u'SimHei',size=13)
plt.rcParams['axes.unicode_minus']=False
plt.title(u"目标函数与样本点")
plt.show()


x0=np.linspace(-3,3,100)
y0=myfun(x0)
plt.plot(x0,y0,color='red',label='目标函数')#目标函数可视化
plt.legend()
plt.show()
# 再用转化的线性回归模型来完成拟合
from sklearn.linear_model import LinearRegression
x_train=x_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)
x_test=np.linspace(-2.5,2.5,6).reshape(-1,1)#测试的样本点

model=LinearRegression()#构建模型
model.fit(x_train,y_train)#模型的训练
print('---线性回归模型---')
print("训练集预测值与样本的残差均方值："+str(np.mean((model.predict(x_train)-y_train)**2)))
print("测试集预测值与目标函数的的残差均方值："+str(np.mean((model.predict(x_test)-myfun(x_test))**2)))
# plt.subplots(2,2,2)
plt.scatter(x_train,y_train,color='green',linewidths=2,label='样本点')#样本点的可视化
plt.legend()
x0=x0.reshape(-1,1)
plt.plot(x0,y0,color='red',label='目标函数')#目标函数可视化
plt.legend()
y1=model.predict(x0)
plt.plot(x0,y1,"b--",label="拟合的直线")#拟合直线的可视化
plt.legend()
plt.show()

# 最后对测试集进行预测。

from sklearn.preprocessing import PolynomialFeatures
featurizer_3=PolynomialFeatures(degree=3)
x_train_3=featurizer_3.fit_transform(x_train)
featurizer_3.get_feature_names()
x_test_3=featurizer_3.fit_transform(x_test)

# 再用转化的线性回归模型来完成拟合
model_3=LinearRegression()
model_3.fit(x_train_3,y_train)

print('---三次多项式回归模型---')
print("训练集预测值与样本的残差均方值："+str(np.mean((model_3.predict(x_train_3)-y_train)**2)))
print("测试集预测值与目标函数的的残差均方值："+str(np.mean((model_3.predict(x_test_3)-myfun(x_test_3))**2)))
y3=model_3.predict(featurizer_3.fit_transform(x0))
plt.plot(x0,y3,'g--',label="三次项拟合")
plt.legend()
plt.show()


