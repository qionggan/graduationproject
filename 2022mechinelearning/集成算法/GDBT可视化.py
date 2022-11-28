import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import graphviz
import matplotlib.pyplot as plt
X = np.arange(1, 11).reshape(-1, 1)
y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])
gbdt = GradientBoostingRegressor(max_depth=4,criterion ='mse').fit(X, y)

# 预测
X_test = np.arange(1, 11, 0.01)[:, np.newaxis]
y_1 = gbdt.predict(X_test)
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=4", linewidth=2)

from sklearn.tree import export_graphviz
# 拟合训练2棵树
sub_tree = gbdt.estimators_[1, 0]
dot_data = export_graphviz(sub_tree, out_file=None, filled=True, rounded=True, special_characters=True, precision=2)
# graph = pydotplus.graph_from_dot_data(dot_data)
graph = graphviz.Source(dot_data)
graph.render("estimator：3")

# Image(graph.create_png())