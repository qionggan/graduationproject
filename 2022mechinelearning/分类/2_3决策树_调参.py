# 决策树调参

# 导入库
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd

# 导入数据集
X = datasets.load_iris()  # 以全部字典形式返回,有data,target,target_names三个键
data = X.data
target = X.target
name = X.target_names
x, y = datasets.load_iris(return_X_y=True)  # 能一次性取前2个
# print(x.shape, y.shape)
# 数据分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=100)
# 用GridSearchCV寻找最优参数（字典）
param = {
    'criterion': ['gini'],
    'max_depth': [30, 50, 60, 100],
    'min_samples_leaf': [2, 3, 10],
    'min_impurity_decrease': [0.1, 0.2, 0.5]#参数搜索范围以字典的形式创建
}
grid = GridSearchCV(DecisionTreeClassifier(), param_grid=param, cv=3)#构建网格搜索模型+6折交叉验证
grid.fit(x_train, y_train)#模型训练
print(pd.DataFrame(grid.cv_results_))#打印每次迭代交叉验证的结果
print('最优分类器:', grid.best_params_, '最优分数:', grid.best_score_)  # 得到最优的参数和分值