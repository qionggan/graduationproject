import numpy as np
from sklearn.linear_model import Ridge
temperature=[15,20,25,30,35,40]
t=np.array(temperature)
humidity=[25,35,45,55,65,75]
h=np.array(humidity)
flower=[136,140,155,160,157,175]
f=np.array(flower)
np.cov(t,f)#温度向量与小花数量向量的协方差矩阵
np.corrcoef(t,f)#温度与小花的数量的相关系数
np.cov(t,h)
np.corrcoef(t,h)#温度与湿度相关系数
new_temperature=[18,22,33]
new_humidity=[31,39,61]
X=np.mat([temperature,humidity])
Y=np.mat(flower)
clf=Ridge(alpha=1.0)#模型构建
clf.fit(X.T,Y.T)#模型的训练
clf.predict(np.mat([new_temperature,new_humidity]).T)#模型的预测
