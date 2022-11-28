import numpy as np
from sklearn.linear_model import  Ridge
temperature=[15,20,25,30,35,40]
t=np.array(temperature)
humidity=[25,35,45,55,65,75]
h=np.array(humidity)
flower=[136,140,155,160,157,175]
f=np.array(flower)
np.cov(t,f)#温度向量与小花数量向量的协方差矩阵
np.corrcoef(t,f)#温度向量与小花数量向量的相关系数矩阵
np.cov(t,h)#湿度向量与温度向量的协方差矩阵
np.corrcoef(t,h)#湿度向量与温度向量的相关系数矩阵
new_temperature=[18,22,33]
new_humidity=[31,39,61]
x_1=temperature
x_2=humidity
X=np.mat([x_1,x_2]) #X=[x1,x2]
Y=np.mat(flower)
clf=Ridge(alpha=1.0)
clf.fit(X.T,Y.T)
pro_num=clf.predict(np.mat([new_temperature,new_humidity]).T)
