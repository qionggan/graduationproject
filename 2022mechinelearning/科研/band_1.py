import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
m=10#可定义m取值，取值为零对应S的半径为零

# 定义本征E的函数表达式fx
def fx(x, y):
    return np.sqrt((np.sqrt(x**2+y**2)-m)**2)
# 画三维能带曲面图
fig = Axes3D(plt.figure())  # 将画布设置为3D
axis_x = np.linspace(-20, 20, 40)  # 设置X轴取值范围
axis_y = np.linspace(-20, 20, 40)  # 设置Y轴取值范围
axis_x, axis_y = np.meshgrid(axis_x, axis_y)  # 将数据转化为网格数据
z = fx(axis_x, axis_y)  # 计算Z轴数值
z_0=np.zeros([40,40])
fig.set_xlabel('X', fontsize=14)
fig.set_ylabel('Y', fontsize=14)
fig.set_zlabel('Z', fontsize=14)
fig.view_init(elev=60, azim=300)  # 设置3D图的俯视角度，方便查看
fig.plot_wireframe(axis_x, axis_y, z, rstride=1, cstride=1,color="r",alpha=0.3,cmap=plt.get_cmap('rainbow'))  # 作出底图
fig.plot_wireframe(axis_x, axis_y, -z, rstride=1, cstride=1,color='b',alpha=0.4,cmap=plt.get_cmap('rainbow'))
fig.plot_surface(axis_x, axis_y, z_0, rstride=1, cstride=1,color='g', cmap=plt.get_cmap('rainbow'))
plt.show()
