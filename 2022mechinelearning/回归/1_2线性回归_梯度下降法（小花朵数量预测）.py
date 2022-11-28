import numpy as np


def gradient(x, y, w):  # 定义梯度函数
    """计算损失函数一阶导数的值
   :param x: 矩阵，样本集
   :param y:矩阵，标签
   :param w:矩阵，线性回归模型的回归系数
   :return:矩阵g，一阶导数值
   """
    m, n = np.shape(x)  # m:样品的个数，n:特征个数
    g = np.mat(np.zeros((n, 1)))

    for j in range(n):
        for i in range(m):
            g[j, 0] -= (y[i, 0] - x[i, :] * w[:, 0]) * x[i, j]
    return g

    # for i in range(m):
    #    for j in range(n):
    #       err = y[i, 0] - x[i, :] * w[:, 0]
    #       g[j,0]-=err*x[i,j]
    # return g


def lossValue(x, y, w):
    """
   :param x:样本集
   :param y:样本标签值
   :param w:回归系数
   :return: 损失函数的值
   """
    l = y - x * w
    return l.T * l / 2


temperatures = [15, 20, 25, 30, 35, 40]
flowers = [136, 140, 155, 160, 157, 175]
x_0 = np.ones(6)
x = np.mat([x_0, temperatures]).T
x.shape
y = np.mat(flowers).T
w = np.mat([0, 0]).T  # 初始w值
alpha = 0.00025  # 学习率
loss_change = 0.000001  # 定义阈值
loss = lossValue(x, y, w)
iteration = 30000
for i in range(iteration):
    w = w - alpha * gradient(x, y, w)  # w的跌代关系
    newloss = lossValue(x, y, w)
    print(str(i) + ":" + str(w[0]) + str(w[1]))
    print(newloss)
    if (abs(loss - newloss) < loss_change):  # 跌代结束判断
        break
    loss = newloss
print(w)

new_temperature = [18, 22, 33]
new_temperature = np.mat([np.ones(3), new_temperature]).T  # 包含x_0=[1,1,1]
pro_num = new_temperature * w  # 预测
print(pro_num)
