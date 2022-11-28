import matplotlib.pyplot as plt
import numpy as np

temperature=[15,20,25,30,35,40]
# temperature=np.linspace(10,35,6)
flower=[136,140,155,160,157,175]
humidity=[0.25,0.35,0.45,0.55,0.65,0.75]
# plt.rcParams['font.sans-serif']=[u'simHei'] #设置字体为SimHei(黑体)
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# plt.scatter(temperature,flower,color='red',label='小花的数量',linewidth=2)
# plt.plot(temperature,flower,linewidth=1)
# plt.xlabel('温度')
# plt.ylabel('花朵的数量')
# plt.title('花朵数量随温度的变化趋势')
# plt.legend()
# plt.grid()
# plt.show()
#method1：最小二乘法
def least_square(X,Y):  #定义最小二乘法函数
    '''
    :param X:矩阵，样本特征矩阵
    :param Y: 矩阵，标签的向量
    :return: W,矩阵，回归系数
    '''
    W=(X*X.T).I*X*Y.T
    return W
# x_0=np.ones(6)
x_0=[1,1,1,1,1,1]
x_1=temperature
x_2=humidity
X=np.mat([x_0,x_1,x_2]) #X=[x0,x1],x0=[1,1,1,1,1,1]
X.shape
# X_I=0.5*np.eye(3)
Y=np.mat(flower)  #Y=[y]
Y.shape
# W=(X*X.T).I

# W=(X*X.T+X_I).I*X*Y.T
# W=least_square(X,Y) #W=[WO,W1].T
# W.shape
# '''
# y=1.43*x+121.56
# '''
# x1=np.linspace(10,35,100)
# y1=W[1,0]*x1+W[0,0]
#
# plt.plot(x1,y1,color='green',label='拟合直线',linestyle='--')
# plt.legend()

# 预测小花数量
new_temperature=[18,22,33]
new_humidity=[0.31,0.39,0.61]
# pro_flower=np.mat(W[2,0]*new_humidity+W[1,0]*new_temperature+W[0,0]).T
# print('%.2f'%pro_flower[1])#保留两位输出
# for i in range(len(pro_flower)):
#     pro_flower[i]=int(pro_flower[i])#整数输出

#
"""
# method2：直接调用sklearn.linear_model子模块当中的LinearRegression 类
"""
from sklearn.linear_model import LinearRegression
lrg=LinearRegression()      #实例化创建对象
lrg.fit(np.mat([temperature,humidity]).T,np.mat(flower).T)  #模型的训练
lrg.coef_ #回归系数
lrg.intercept_#截距
lrg.predict(np.mat([new_temperature,new_humidity]).T) #模型进行预测