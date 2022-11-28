# 这是双参数下的一个梯度下降过程可视化图形
# 梯度函数

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def Func_J(x, y):
    return 2 * np.power(x, 2) + np.power(y, 2)


# 梯度函数的导数
def grad_theta_1(theta):
    return 4 * theta


def grad_theta_2(theta):
    return 2 * theta


def gradient_descent(eta, n_iters, theta1, theta2, up, dirc):
    t1 = [theta1]
    t2 = [theta2]
    for i in range(n_iters):
        gradient = grad_theta_1(theta1)
        theta1 = theta1 - eta * gradient
        t1.append(theta1)
        gradient = grad_theta_2(theta2)
        theta2 = theta2 - eta * gradient
        t2.append(theta2)

    plt.figure(figsize=(10, 10))  # 设置画布大小
    # 生成虚拟数据
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)

    # 转换成网格数据
    X, Y = np.meshgrid(x, y)
    Z = Func_J(X, Y)
    ax = plt.axes(projection='3d')
    fig = plt.figure()
    ax.contour3D(X, Y, Func_J(X, Y), 50, cmap='binary')  # 等高线图

    ax.scatter3D(t1, t2, Func_J(t1, t2), c='r', marker='o')

    ax.view_init(up, dirc)

    return t1, t2
plt.show()

# %matplotlib inline


@interact(eta=(0, 2, 0.0002), n_iters=(1, 100, 1), initial_theta1=(-3, 3, 0.1), initial_theta2=(-3, 3, 0.1),
          up=(-180, 180, 1), dirc=(-180, 180, 1), continuous_update=False)
# lr为学习率（步长） epoch为迭代次数   init_theta为初始参数的设置 up调整图片上下视角 dirc调整左右视角
def visualize_gradient_descent(eta=0.05, n_iters=10, initial_theta1=-2, initial_theta2=-3, up=45, dirc=100):
    gradient_descent(eta, n_iters, initial_theta1, initial_theta2, up, dirc)