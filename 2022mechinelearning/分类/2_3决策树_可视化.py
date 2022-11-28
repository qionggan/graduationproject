from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import numpy as np
# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    data = np.array(df.iloc[:100, [0, 1, -1]])
    # print(data)
    return data[:, :2], data[:, -1],iris.feature_names[0:2]


X, y,feature_name= create_data()
data = np.loadtxt('data1.txt')
data.shape
X = data[:, 0:-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn import tree
# 模型训练
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)#精度

tree.plot_tree(clf,filled=True)#决策树可视化

#使用函数 export_text以文本格式导出树
from sklearn.tree import export_text
# r = export_text(clf,feature_name)
r = export_text(clf)
print(r)

# 导出树
dot_data = tree.export_graphviz(clf, out_file=None,filled=True)
graph = graphviz.Source(dot_data)
graph.render("iris") #保存为pdf格式



