import matplotlib.pyplot as plt
err_train=[2019,534,209,4]
err_test=[578,247,1232,3848]
x_n=[1,3,5,9]
plt.scatter(x_n,err_test,label='测试误差',linewidths=3)
plt.plot(x_n,err_test)
plt.scatter(x_n,err_train,label='训练误差',linewidths=3)
plt.plot(x_n,err_train)
plt.xlim(0,10)
plt.rc('font',family=u'SimHei',size=20)
plt.legend(loc="best")
plt.show()
plt.grid()
