import matplotlib.pyplot as plt
import numpy as np
temperature=[10,15,20,25,30,35]
# temperature=np.linspace(10,35,6)
flower=[136,140,155,160,157,175]
plt.rcParams["font.sans-serif"]=[u'simHei']
plt.scatter(temperature,flower,color='red',label="小花的数量",linewidths=2)
plt.plot(temperature,flower,linewidth=1)
plt.legend()
plt.ylabel('花朵的数量')
plt.xlabel('温度')
plt.grid()
plt.title('花朵的数量随着温度的变化趋势')
plt.show()


def least_square(X,Y):#最小二乘法函数定义
    '''
    :param X: 矩阵，样本的特征矩阵
    :param Y: 样本标签向量
    :return: W 回归系数
    '''
    W=(X*X.T).I*X*Y.T
    return W
# x_0=[1,1,1,1,1,1]
x_0=np.ones(6)
x_1=temperature
X=np.mat([x_0,x_1])
X.shape
print(X)
Y=np.mat(flower)
Y.shape
W=least_square(X,Y)
print(W)
W.shape

#y=1.43*x+121
x1=np.linspace(10,35,100)
y1=W[1,0]*x1+W[0,0]
plt.plot(x1,y1,color='green',label='拟合的直线',linestyle='--')
plt.legend()
plt.show()

#预测小花的数量
new_temperature=np.mat([40,22,33])
pro_flower=(W[1,0]*new_temperature+W[0,0]).T
print(pro_flower)

for i in range(len(pro_flower)):
    pro_flower[i]=int(pro_flower[i])
print(pro_flower)


# 方法2:直接调用sklearn.linear_model
from sklearn.linear_model import LinearRegression
lrg=LinearRegression()#实例化创建对象
lrg.fit(np.mat(temperature).T,np.mat(flower).T)#模型的训练
lrg.coef_
lrg.intercept_
#训练出的模型为：y=1.43*x+121
lrg.predict(np.mat(new_temperature).T)#模型的预测