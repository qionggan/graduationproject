import numpy as np
list1=[143.7,162.2,199.5,96.5]
list2=[31,31,10,31]
list3=[10,8,10,13]
list4=[105,118,170,74]
data=np.array([list1,list2,list3,list4])
# data=np.array(data).astype(float)
shape=data.shape
# 数据标准化
for i in range(shape[1]):
#获取该类每行数据的均值和标准差
    mean=np.mean(data[:,i])
    std=np.std(data[:,i])
    for j in range(shape[0]):
        data[j,i]=(data[j,i]-mean)/std
print((data).T)

# 数据归一化
# for i in range(shape[0]):
# #获取该类数据没列的最大值和最小值
#     max=np.max(data[i,:])
#     min=np.min(data[i,:])
#     for j in range(shape[1]):
#         data[i,j]=(data[i,j]-min)/(max-min)
# print((data).T)

# 调用sklearn中preprocessing子模块
from sklearn import preprocessing
standard_scaler_data = preprocessing.StandardScaler().fit_transform(data)


# min_max_scaler_data = preprocessing.MinMaxScaler().fit_transform((data).T)


