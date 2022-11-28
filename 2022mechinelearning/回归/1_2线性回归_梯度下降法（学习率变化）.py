import matplotlib.pyplot as plt
import numpy as np
x_data=[338.,333.,328.,207.,226.,25.,179.,60.,208.,606.]
y_data=[640.,633.,619.,393.,428.,27.,193.,66.,226,1591]
#y_data=b+w*x_data
plt.scatter(x_data,y_data)

#损失函数的计算
x=np.arange(-200,-100,1) #截距b
y=np.arange(-5,5,0.1)#权重w
z=np.zeros((len(x),len(y)))
X,Y=np.meshgrid(x,y)
for i in range(len(x)):
    for j in range(len(y)):
        b=x[i]
        w=y[i]
        z[j][i]=0
        for n in range(len(x_data)):
            z[j][i]=z[j][i]+y_data[n]-b-w*x_data[n]**2
        z[j][i]=z[j][i]/len(x_data)
print(z)

#y_data=b+w*x_data
b=-140#初始点
w=-4 #初始点
#lr=0.0000001 #学习率
# lr=0.000001 #学习率
lr=1
iteration=100000#跌代的次数
#将迭代过程中的b，w进行保存，以便后面画图
w_history=[w]
b_history=[b]
lr_b=0
lr_w=0
#迭代
for i in range(iteration):
    b_grad=0.0
    w_grad=0.0
    for n in range(len(x_data)):
        b_grad=b_grad-2.0*(y_data[n]-b-w*x_data[n]*1.0)   #∑(yi-f(xi))*xi      损失函数对b的求导
        w_grad=w_grad-2.0*(y_data[n]-b-w*x_data[n])*x_data[n] #损失函数对w的求导

    # 更新参数  adagradient 学习率自适应调整
    lr_b=lr_b+b_grad**2
    lr_w=lr_w+w_grad**2
    b = b - lr/np.sqrt(lr_b) * b_grad
    w = w - lr/np.sqrt(lr_w)* w_grad

    # 更新参数  学习率为lr
    # b=b-lr*b_grad
    # w=w-lr*w_grad

    # 保存b和w的历史数据
    b_history.append(b)
    w_history.append(w)
#可视化
plt.contourf(x,y,z,50,cmap=plt.get_cmap('jet'))
plt.plot([-188.4],[2.67],'*',ms=12,markeredgewidth=3,color='orange')
plt.plot(b_history,w_history,'o-',ms=3,lw=1.5,color='black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$b$',fontsize=16)
plt.ylabel(r'$w$',fontsize=16)


