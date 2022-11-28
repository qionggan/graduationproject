import matplotlib.pyplot as plt
import numpy as np
# 先拟定一个一元三次多项式作为目标函数
def myfun(x):
    '''目标函数
    input:x(float):自变量
    output:函数值'''
    return 10 + 5 * x + 4 * x**2 + 6 * x**3
# 构造一批样本点
x_train = np.linspace(-3,3, 15)
import random
y_train = myfun(x_train) + np.random.random(size=len(x_train)) * 100 - 50#加上一些噪声产生样本集

plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rc('font', family='SimHei', size=13)#字体设置为黑体
# plt.scatter(x_train, y_train, color="green“, linewidth=2,label='样本点')#样本点可视化
# plt.legend()
# plt.title(u'目标函数与测试样本点')

x0= np.linspace(-3, 3, 100)
y0 = myfun(x0)
# plt.plot(x0, y0, color="red", linewidth=1,label='目标函数')#目标函数可视化
# plt.legend()
# plt.show()


from sklearn.linear_model import LinearRegression
import numpy as np
x_train = x_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)
x_test = (np.linspace(-2.5, 2.5, 6)).reshape(-1,1) # 预测样本点
#线性回归模型
model = LinearRegression()# 创建模型
model.fit(x_train, y_train)# 模型训练
print('--线性回归模型--')
print('训练集预测值与样本的残差均方值：' + str(np.mean((model.predict(x_train)-y_train)**2)))
print('测试集预测值与目标函数值的残差均方值：' + str(np.mean((model.predict(x_test)-myfun(x_test))**2)))
print('系数：' + str(model.coef_))
plt.subplot(2, 2, 1)
plt.title(u'线性回归模型预测')
plt.scatter(x_train, y_train, color="green", linewidth=2,label='样本点')#样本点可视化
plt.legend()
x0 = x0.reshape(-1,1)
plt.plot(x0, y0, color="red", linewidth=1,label='目标函数')#目标函数可视化
plt.legend()
y1 = model.predict(x0)
plt.plot(x0, y1, "b--", linewidth=1,label='拟合的直线')#拟合的直线可视化
plt.legend()
plt.show()

#三次多项式模型
from sklearn.preprocessing import PolynomialFeatures
featurizer_3 = PolynomialFeatures(degree=3)
x_train_3 = featurizer_3.fit_transform(x_train)  #x_train特征映射高阶项
featurizer_3.get_feature_names()
x_test_3 = featurizer_3.transform(x_test)#x_test特征映射高阶项
x_test_3
# 再用转化的线性回归模型来完成拟合
model_3 = LinearRegression()
model_3.fit(x_train_3, y_train)
print('--三次多项式模型--')
print('训练集预测值与样本的残差均方值：' + str(np.mean((model_3.predict(x_train_3)-y_train)**2)))
print('测试集预测值与目标函数值的残差均方值：' + str(np.mean((model_3.predict(x_test_3)-myfun(x_test))**2)))
print('系数：' + str(model_3.coef_))

plt.subplot(2, 2, 2)
plt.title(u'三次多项式模型预测')
plt.scatter(x_train, y_train, color="green", linewidth=2,label='样本点')
plt.legend()
plt.plot(x0, y0, color="red", linewidth=1,label='目标函数')
plt.legend()
#y1 = model.predict(x1)
#plt.plot(x1, y1, color="black", linewidth=1)
y3 = model_3.predict(featurizer_3.fit_transform(x0))
plt.plot(x0, y3, "b--", linewidth=1,label='三次项拟合')
plt.legend()
plt.show()

plt.subplot(2, 2, 3)
#五次多项式模型
from sklearn.preprocessing import PolynomialFeatures
featurizer_5 = PolynomialFeatures(degree=5)
x_5 = featurizer_5.fit_transform(x_train)
x_p_5 = featurizer_5.transform(x_test)
model_5 = LinearRegression()
model_5.fit(x_5, y_train)
print('--五次多项式模型--')
print('训练集预测值与样本的残差均方值：' + str(np.mean((model_5.predict(x_5)-y_train)**2)))
print('测试集预测值与目标函数值的残差均方值：' + str(np.mean((model_5.predict(x_p_5)-myfun(x_test))**2)))
print('系数：' + str(model_5.coef_))

plt.title(u'五次多项式模型预测')
plt.scatter(x_train, y_train, color="green", linewidth=2)
plt.plot(x0, y0, color="red", linewidth=1)
#y1 = model.predict(x1)
#plt.plot(x1, y1, color="black", linewidth=1)
#y3 = model_3.predict(featurizer_3.fit_transform(x1))
#plt.plot(x1, y3, "b--", linewidth=1)
y5 = model_5.predict(featurizer_5.fit_transform(x0))
plt.plot(x0, y5, "b--", linewidth=1,label='五次项模型拟合')
plt.legend()
plt.show()

plt.subplot(2, 2, 4)
#九次多项式模型
from sklearn.preprocessing import PolynomialFeatures
featurizer_9 = PolynomialFeatures(degree=12)
x_9 = featurizer_9.fit_transform(x_train)
x_p_9 = featurizer_9.transform(x_test)
model_9 = LinearRegression()
model_9.fit(x_9, y_train)
print('--九次多项式模型--')
print('训练集预测值与样本的残差均方值：' + str(np.mean((model_9.predict(x_9)-y_train)**2)))
print('测试集预测值与目标函数值的残差均方值：' + str(np.mean((model_9.predict(x_p_9)-myfun(x_test))**2)))
print('系数：' + str(model_9.coef_))

plt.title(u'九次多项式模型预测')
plt.scatter(x_train, y_train, color="green", linewidth=2)
plt.plot(x0, y0, color="red", linewidth=1)
#y1 = model.predict(x1)
#plt.plot(x1, y1, color="black", linewidth=1)
#y3 = model_3.predict(featurizer_3.fit_transform(x1))
#plt.plot(x1, y3, "b--", linewidth=1)
y9 = model_9.predict(featurizer_9.fit_transform(x0))
plt.plot(x0, y9, "c--", linewidth=1,label='九次项拟合')
plt.legend()
plt.show()