import matplotlib.pyplot as plt
import numpy as np
'''
最小二乘法直线拟合
'''
voltage=[0.74,1.52,2.33,3.08,3.66,4.49]
electric_current=[2.00,4.01,6.22,8.20,9.75,12.00]
plt.rcParams['font.sans-serif']=[u'simHei'] #设置字体为SimHei(黑体)
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

plt.scatter(voltage,electric_current,color='red',label='实验数据点',linewidth=2)
plt.plot(voltage,electric_current,linewidth=1)
plt.xlabel('电压/V',fontsize=20)
plt.ylabel('电流/mA',fontsize=20)
plt.title('光敏电阻伏安曲线',fontsize=20)
plt.legend(fontsize=20)
plt.grid()
plt.show()
# 1.定义最小二乘法函数
def least_square(X,Y):
    '''
    :param X:矩阵，样本特征矩阵
    :param Y: 矩阵，标签的向量
    :return: W,矩阵，回归系数
    '''
    W=(X*X.T).I*X*Y.T
    return W
# 2. 调用最小二乘法函数求解直线方程参数W
x_0=np.ones(6)
x_1=voltage
X=np.mat([x_0,x_1]) #X=[x0,x1],x0=[1,1,1,1,1,1]
X.shape
Y=np.mat(electric_current)  #Y=[y]
Y.shape
W=least_square(X,Y) #W=[WO,W1].T
W.shape
'''
拟合直线方程为：y=1.43*x+121.56
'''
print(W)
#3. 绘制拟合直线
x1=np.linspace(0,5,50)
y1=W[1,0]*x1+W[0,0]
plt.plot(x1,y1,color='green',label='拟合直线',linestyle='--')
plt.legend(fontsize=20)
