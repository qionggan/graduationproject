import numpy as np
def f(x_1, x_2):
    '''给平面上的数据点打标签，直线上面的点标签为1，否则为0
    para x: float, x值
    para y: float, y值
    return: int 0-1标签值'''
    yp = 2.0 * x_1 - 30
    if x_2 > yp:
        return 1
    else:
        return 0

import random
# np.random.seed(42) #设置随机数种子
samples = []
for i in range(0, 100):
    x_1 = random.uniform(0, 100)
    x_2 = random.uniform(0, 100)
    samples.append([x_1, x_2, f(x_1, x_2)])
samples
np.savetxt('sample.txt',samples) #数据保存为sample.txt

# 随机产生点的可视化
import matplotlib.pyplot as plt
x1_1 = []
x1_2 = []
x0_1 = []
x0_2 = []
for x in samples:
    if x[-1] > 0.0:
        x1_1.append(x[0])
        x1_2.append(x[1])
    else:
        x0_1.append(x[0])
        x0_2.append(x[1])
plt.scatter(x1_1, x1_2, c='r', marker='.')
plt.scatter(x0_1, x0_2, c='b', marker='x')
plt.show()

#定义代表虚线的函数
def fw(W,x):
    # return (W * x)[0,0]
    return np.dot(W, x)

import numpy as np
#sigmoid函数
def g(z):
    return(1 / (1 + np.exp(-z)))

#损失函数
def lossValue(W, x, y):
    LW = 0.0
    for i in range(len(y)):
        hw = g(fw(W, np.mat(x[:,i]).T))
        LW = LW + y[i] * np.log(hw) + (1 - y[i]) * np.log(1 - hw)
    if np.isnan(LW):
        return(np.inf)
    return -1.0 * LW

#梯度函数定义
def gradient(x, y, W):
    n, m = np.shape(x)
    gg = np.mat(np.zeros((n, 1)))
    for j in range(n):
        err = 0.0
        for i in range(m):
            hw = g(fw(W, np.mat(x[:,i]).T))
            err = err + x[j,i] * (hw - y[i]) #x_i*(g(z)-y_i)
        gg[j] = err
    print(gg)
    return gg

# 梯度下降法求解W
W = np.mat([0.0,0.0,0.0])
samples = np.array(samples)
samples.shape
x = np.c_[np.ones((samples.shape[0],1)), samples[:, 0:2]]#按列连接矩阵
x = x.T
y = samples[:,-1]
alpha = 0.00001 #学习率
loss_change = 0.000001#迭代最小的阈值
loss = lossValue(W, x, y)
print(loss)
for i in range(50000):
    if i < 15000:
        alpha = 0.0001
    elif i <30000:
        alpha = 0.00005
    elif i < 40000:
        alpha = 0.00001
    else:
        alpha = 0.000005
    W = W - alpha * gradient(x, y, W).T #梯度迭代关系式
    newloss = lossValue(W, x, y)
    print(str(i)+":: "+str(W[0,0])+' : '+str(W[0,1])+' : '+str(W[0,2]))
    print(newloss)
    if abs(loss - newloss) < loss_change:
        break
    loss = newloss
print(W)

# 决策函数(边界)可视化  w2*x2+w1*x1+w0*x0=0
tx_1 = np.linspace(15,65,100)
w1 = W[0,1] / W[0,2]
w0 = W[0,0] / W[0,2]
tx_2 = - w1 * tx_1 - w0
plt.plot(tx_1, tx_2, color="black", linewidth=2, linestyle="--" )
plt.show()
W
# 预测函数定义
def predict(X, W):
    m = X.shape[0]  #样本的个数
    p = np.zeros((m, 1))   #初始化为0
    p = g(np.dot(X, W))  # 预测的结果，是个概率值
    for i in range(m):
        if p[i] > 0.5:  # 概率大于0.5预测为1，否则预测为0
            p[i] = 1
        else:
            p[i] = 0
    return p
# 新的样本点预测
X1_new=[80,10,76,15,67]
X2_new=[20,70,9,40,85]
X0_new=[1,1,1,1,1]
samples_new=np.array([X0_new,X1_new,X2_new]).T
res=predict(samples_new,W.T)


# -----------method2：直接调用sklearn中的Logisticsregression 类-----
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('sample.txt')
X = data[:, 0:-1]
y = data[:, -1]
# samples = np.array(samples)
# X = samples[:, 0:-1]
# y = samples[:, -1]

# 划分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=20)
# 归一化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# 逻辑回归
def plot_decision_boundary(model,axis):
    x0,x1=np.meshgrid(
        np.linspace(axis[0],axis[1],500).reshape(-1,1),
        np.linspace(axis[2],axis[3],500).reshape(-1,1)
    )
    x_new=np.c_[x0.ravel(),x1.ravel()]
    y_pre=model.predict(x_new)
    zz=y_pre.reshape(x0.shape)
    # from matplotlib.colors import ListedColormap
    # cus=ListedColormap(["#EF9A9A","#FFF59D","#90CAF9"])
    # plt.contourf(x0,x1,zz,cmap=cus)
    plt.contour(x0, x1, zz, [0.0, 0.01], linewidth=2.0)  # 画等高线，范围在[0,0.01]

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
plot_decision_boundary(model,axis=[0,100,0,100])

# x_test预测
predict = model.predict(x_test)
print(model.score(x_train,y_train))
print(model.score(x_test,y_test))

# 新样本的预测
X1_new=[80,10,76,15,67]
X2_new=[20,70,9,40,85]
samples_new=np.array([X1_new,X2_new]).T
res=model.predict(samples_new)
res

x1_1 = []
x1_2 = []
x0_1 = []
x0_2 = []
for x in data:
    if x[-1] > 0.0:
        x1_1.append(x[0])
        x1_2.append(x[1])
    else:
        x0_1.append(x[0])
        x0_2.append(x[1])
plt.scatter(x1_1, x1_2, c='r', marker='.')
plt.scatter(x0_1, x0_2, c='b', marker='x')
plt.show()


