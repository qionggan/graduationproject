import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
phi=np.pi/4
q_1=0
q_2=0
# 定义本征值E的函数表达式fx
def fx(x, y):
    if np.cos(2*phi)>0:
        z_1 =(x - q_1) * np.tan(2 * phi)+np.sqrt((x-q_1)**2/(np.cos(2*phi))**2+(y-q_2)**2/np.cos(2*phi))
        z_2 =(x - q_1) * np.tan(2 * phi)-np.sqrt((x-q_1)**2/(np.cos(2*phi))**2+(y-q_2)**2/np.cos(2*phi))
    z_1 = (x - q_1) * np.tan(2 * phi)+ np.sqrt((x-q_1) ** 2 / (np.cos(2 * phi)) ** 2 + (y-q_2) ** 2 /(- np.cos(2 * phi)))
    z_2 = (x - q_1) * np.tan(2 * phi)- np.sqrt((x-q_1) ** 2 / (np.cos(2 * phi)) ** 2 + (y-q_2) ** 2 /(- np.cos(2 * phi)))
    return z_1,z_2
# 画三维能带曲面图
fig = Axes3D(plt.figure())  # 将画布设置为3D
axis_x = np.linspace(-5, 5, 100)  # 设置X轴取值范围
axis_y = np.linspace(-5, 5, 100)  # 设置Y轴取值范围
axis_x, axis_y = np.meshgrid(axis_x, axis_y)  # 将数据转化为网格数据
z_1,z_2 = fx(axis_x, axis_y)  # 计算Z轴数值
z_0=np.zeros([100,100])
fig.set_xlabel('qa', fontsize=14)
fig.set_ylabel('qb', fontsize=14)
fig.set_zlabel('E', fontsize=14)
fig.view_init(elev=60, azim=300)  # 设置3D图的俯视角度，方便查看
fig.plot_wireframe(axis_x, axis_y, z_1, rstride=3, cstride=1,alpha=0.3,color='r')  # 作出底图
fig.plot_wireframe(axis_x, axis_y, z_2, rstride=3, cstride=1,alpha=0.53,color='b',)
fig.plot_surface(axis_x, axis_y, z_0, rstride=1, cstride=1,color='green',alpha=0.3)
plt.show()
