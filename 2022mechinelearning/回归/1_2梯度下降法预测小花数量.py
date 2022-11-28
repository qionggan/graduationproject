import numpy as np
def gradient(x,y,w):#定义梯度函数
    """

    :param x: 矩阵，样本集
    :param y: 矩阵，标签
    :param w: 矩阵，回归系数
    :return: g，一阶导数值，梯度
    """
    m,n=np.shape(x) #m表示样本的个数，n表示特征个数
    g=np.mat(np.zeros((n,1)))#梯度值初始化矩阵为零（n行1列）
    for j in range(n):
        for i in range(m):
            g[j,0]-=(y[i,0]-x[i,:]*w[:,0])*x[i,j]#梯度值
    return g

#定义损失函数
def lossvalue(x,y,w):
    l=y-x*w
    return l.T*l/2

temperature=[10,15,20,25,30,35]
flower=[136,140,155,160,157,175]
x_0=np.ones(6)
x=np.mat([x_0,temperature]).T
y=np.mat(flower).T
w=np.mat([0,0]).T
alpha=0.00025
loss_change=0.000001
loss=lossvalue(x,y,w)
iteration=30000
for i in range(iteration):
    w=w-alpha*gradient(x,y,w)#梯度下降法迭代关系式
    newloss=lossvalue(x,y,w)
    print(str(i)+':'+str(w[0])+str(w[1]))
    print(newloss)
    if(abs(loss-newloss)<loss_change):
        break
    loss=newloss
#新的损失函数值输出
print(w)
#w[0]=121,w[1]=1.45
#线性模型为：y=w[1]*x+w[0]

#预测小花数量
new_temperature=[18,20,33]
x_0=np.ones(3)
new_temperature=np.mat([x_0,new_temperature]).T
pro_num=new_temperature*w
pro_num


