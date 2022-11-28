import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 解决字体设置问题
# data = np.loadtxt('linear.txt')
# X = data[:, 0:-1]
# y = data[:, -1]
#
# def plot_data(X, y):
#     pos = np.where(y == 1)  # 找到y==1的坐标位置
#     neg = np.where(y == 0)  # 找到y==0的坐标位置
#     # 作图
#     plt.figure(figsize=(15, 12))
#     plt.plot(X[pos, 0], X[pos, 1], 'ro')  # red o
#     plt.plot(X[neg, 0], X[neg, 1], 'bo')  # blue o
#     plt.title(u"两个类别散点图", fontproperties=font)
#     plt.show()
# plot_data(X, y)
#
#
#
#
#
#
# # x1_1 = []
# # x1_2 = []
# # x0_1 = []
# # x0_2 = []
# # for x in data:
# #     if x[-1] > 0.0:
# #         x1_1.append(x[0])
# #         x1_2.append(x[1])
# #     else:
# #         x0_1.append(x[0])
# #         x0_2.append(x[1])
# # plt.scatter(x1_1, x1_2, c='r', marker='.')
# # plt.scatter(x0_1, x0_2, c='b', marker='x')
# # plt.show()
#
# #代表虚线的函数
# def fw(W,x):
#     return (W * x)[0,0]
#
# import numpy as np
# #sigmoid函数
# def g(z):
#     return(1 / (1 + np.exp(-z)))
#
# #损失函数
# def lossValue(W, x, y):
#     LW = 0.0
#     for i in range(len(y)):
#         #ff = fw(W, np.mat(x[:,i]).T)
#         hw = g(fw(W, np.mat(x[:,i]).T))
#         LW = LW + y[i] * np.log(hw) + (1 - y[i]) * np.log(1 - hw)
#     if np.isnan(LW):
#         return(np.inf)
#     return -1.0 * LW
#
# #梯度
# def gradient(x, y, W):
#     n, m = np.shape(x)
#     gg = np.mat(np.zeros((n, 1)))
#     for j in range(n):
#         err = 0.0
#         for i in range(m):
#             hw = g(fw(W, np.mat(x[:,i]).T))
#             err = err + x[j,i] * (hw - y[i])
#         gg[j] = err
#     print(gg)
#     return gg
#
# # 映射为多项式
# def mapFeature(X1, X2):
#     degree = 2;  # 映射的最高次方
#     out = np.ones((X1.shape[0], 1))  # 映射后的结果数组（取代X）
#     '''
#     这里以degree=2为例，映射为1,x1,x2,x1^2,x1,x2,x2^2
#     '''
#     for i in np.arange(1, degree + 1):
#         for j in range(i + 1):
#             temp = X1 ** (i - j) * (X2 ** j)  # 矩阵直接乘相当于matlab中的点乘.*
#             out = np.hstack((out, temp.reshape(-1, 1)))
#     return out
#
# W = np.mat([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
# samples = np.array(data)
# samples.shape
# samples_map=mapFeature(samples[:,0:1],samples[:,1:2])
# x = np.c_[np.ones((samples.shape[0],1)), samples_map[:, :]]#按列连接
# x = x.T
# x.shape
#
# y = samples[:,-1]
# alpha = 0.00001
# loss_change = 0.000001
# loss = lossValue(W, x, y)
# print(loss)
# for i in range(40000):
#     if i < 15000:
#         alpha = 0.0001
#     elif i <30000:
#         alpha = 0.00005
#     elif i < 40000:
#         alpha = 0.00001
#     else:
#         alpha = 0.000005
#     W = W - alpha * gradient(x, y, W).T
#     newloss = lossValue(W, x, y)
#     print(str(i)+":: "+str(W[0,0])+' : '+str(W[0,1])+' : '+str(W[0,2])+' : '+str(W[0,3])+' : '+str(W[0,4])+' : '+str(W[0,5])+' : '+str(W[0,6]))
#     print(newloss)
#     if abs(loss - newloss) < loss_change:
#         break
#     loss = newloss
# W
# u = np.linspace(0, 100, 100)  # 根据具体的数据，这里需要调整
# v = np.linspace(0, 100, 100)
# u.shape[0]
# z = np.zeros((len(u), len(v)))
# x_0=np.ones((u.shape[0],1))
#
# for i in range(len(u)):
#     for j in range(len(v)):
#         x = np.c_[x_0[i], mapFeature(u[i].reshape(1, -1), v[j].reshape(1, -1))]
#         z[i, j] = fw(W,x.T) # 计算对应的值，需要map
#
# z = np.transpose(z)
# plt.contour(u, v, z, [0, 0.01], linewidth=2.0)  # 画等高线，范围在[0,0.01]，即近似为决策边界
# # plt.legend()
# plt.show()





from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ellipseSamples.txt')

X = data[:, 0:-1]
y = data[:, -1]
# 添加噪声 y[4,100]=0-->1, 1-->0
# # for i in range(4,100):
# #     if y[i]<1:
# #         y[i]=1
# #     else:
# #         y[i] = 0
# 逻辑回归
def plot_decision_boundary(model,axis):
    x0,x1=np.meshgrid(
        np.linspace(axis[0], axis[1],500).reshape(-1, 1),
        np.linspace(axis[2], axis[3],500).reshape(-1, 1)
    )
    x_new=np.c_[x0.ravel(),x1.ravel()]
    y_pre=model.predict(x_new)
    zz=y_pre.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    cus=ListedColormap(["#0000FF","#FF00FF","#FFC0CB"])#蓝色、紫色、粉色
    plt.contourf(x0,x1,zz,cmap=cus)
    # plt.contour(x0,x1,zz, [0, 0.01], linewidth=2.0)  # 画等高线，范围在[0,0.01]

def polynomiallogisticregression(degree):
    return Pipeline([
        ("poly",PolynomialFeatures(degree=degree)),
        ("std_reg",StandardScaler()),
        ("log_reg",LogisticRegression(penalty='l2',C=1))
    ])

# 划分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=20)


p1=polynomiallogisticregression(degree=9)
p1.fit(x_train,y_train)#模型训练
print(p1.score(x_train,y_train))#训练误差
# print(p1.score(x_test,y_test))#测试准确度
plot_decision_boundary(p1,axis=[-1000,1000,-1000,1000])

# 新样本的预测
X1_new=[-999,10,76,15,67]
X2_new=[999,70,9,40,85]
samples_new=np.array([X1_new,X2_new]).T
res=p1.predict(samples_new)
res

# 样本散点图
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
plt.scatter(x0_1, x0_2, c='k', marker='x')
plt.show()
