from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
load = load_boston()
x = load.data
y = load.target
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.3)
idx = []
mse_list = []
loss = ['ls','lad','huber','quantile']
fig = plt.figure(figsize=(18,12))
for i in range(len(loss)):
    for j in range(1,200):
        reg = GradientBoostingRegressor(j,loss=loss[i],max_depth=1,alpha=0.8)
        reg.fit(x_train,y_train)
        pred = reg.predict(x_test)
        error= mse(y_test,pred)
        idx.append(j)
        mse_list.append(error)
    ax = fig.add_subplot(2,2,i+1)
    ax.plot(idx,mse_list)
    ax.set_xlabel('n_estimator')
    ax.set_ylabel('mse')
    ax.set_title(loss[i])
    idx=[]
    mse_list=[]