import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
# 1.导入数据并转化为DateFrame
data_iris=load_iris()
data=pd.DataFrame(load_iris().data,columns=load_iris().feature_names)
target=load_iris().target
data['target']=pd.DataFrame(target)
data.tail(5)
data.columns
# 样本点的可视化
# plt.scatter(data['sepal length (cm)'].values, data['petal length (cm)'].values)
# plt.xlabel('sepal length (cm)',fontsize=20)
# plt.ylabel('petal length (cm)',fontsize=20)

plt.scatter(data['petal width (cm)'].values, data['petal length (cm)'].values)
plt.xlabel('petal width (cm)',fontsize=20)
plt.ylabel('petal length (cm)',fontsize=20)

X = data.iloc[:, 0:3].values
Y = data.iloc[:,-1].values
# 将数据集分成训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
# 特征缩放
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# 2.逻辑回归模型
# 将逻辑回归应用于训练集
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
# 3.预测
# 测试集结果
y_pred = classifier.predict(X_test)
# 4.模型的评估
# 生成混淆矩阵和分类报告
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cm = confusion_matrix(y_test, y_pred)#生成混淆矩阵
res=classification_report(y_test,y_pred)#打印分类报告
print(res)

