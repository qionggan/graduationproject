# 1.导包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# 2.数据集准备
wine = load_wine()
X = wine.data
y= wine.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.3,random_state =42)
# 3.模型建立
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

# 随机森林和决策树在交叉验证下的效果对比
DF_cv = []
RF_cv = []
for i in range(10):#cv遍历10次
    clf=DecisionTreeClassifier()
    rfc = RandomForestClassifier(n_estimators=20)
    DF_cv.append(cross_val_score(clf,X,y,cv=5).mean())
    RF_cv.append(cross_val_score(rfc,X,y,cv=5).mean())

plt.plot(range(1,11),DF_cv,label='DecisionTreeClassifier')
plt.plot(range(1,11),RF_cv,label='RandomForestClassifier')
plt.legend(loc=7)
plt.show();

# n_estimators学习曲线
score= []
for i in range(100):
    rfc = RandomForestClassifier(n_estimators= i+1)
    cv_score = cross_val_score(rfc,X,y,cv=10).mean()
    score.append(cv_score)
plt.figure(figsize=(15,7),dpi=80)
plt.plot(score);
plt.show()


