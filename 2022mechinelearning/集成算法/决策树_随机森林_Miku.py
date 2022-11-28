import numpy as np
import matplotlib.pyplot as plt
import pybaobabdt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

data = np.loadtxt('miku.txt',encoding="utf-8")
data.shape
X = data[:, 0:-1]
y = data[:, -1]
X.min()

def plot_decision_boundary(model,axis):
    x0,x1=np.meshgrid(
        np.linspace(axis[0], axis[1],500).reshape(-1, 1),
        np.linspace(axis[2], axis[3],500).reshape(-1, 1)
    )
    x_new=np.c_[x0.ravel(),x1.ravel()]
    y_pre=model.predict(x_new)
    zz=y_pre.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    cus=ListedColormap(["#0000FF","#FF00FF","#FFC0CB"])#蓝色、紫色、粉色
    plt.contourf(x0,x1,zz,cmap=cus)
for i,j in enumerate([5,10,15,20]):
    # clf = DecisionTreeClassifier(max_depth=j, random_state=42).fit(-X, -y)# 建立并训练模型
    clf = RandomForestClassifier(max_depth=j,n_estimators=20, n_jobs=-1, random_state=42).fit(-X, y)
    plt.subplot(2, 2, i + 1)
    plot_decision_boundary(clf,axis=[-2.5,2.5,-2.5,2.5])
    plt.title(r"$maxdepth = {}$".format(j), fontsize=24)
plt.show()


df = pd.DataFrame(data,columns=["x1","x2","class"])
y = list(df['class'])
features = list(df.columns)
features.remove('class')
X = df.loc[:, features]
clf = DecisionTreeClassifier(max_depth=5).fit(-X, y)# 建立并训练模型
pybaobabdt.drawTree(clf, features=features,dpi=300)
# clf = RandomForestClassifier(max_depth=5,n_estimators=20, n_jobs=-1, random_state=0)
# clf.fit(X, y)
# fig = plt.figure(figsize=(15,15), dpi=300)
# for idx, tree in enumerate(clf.estimators_):
#     ax1 = fig.add_subplot(5, 4, idx + 1)
#     pybaobabdt.drawTree(tree, model=clf, size=15, dpi=300, ax=ax1)
#
# fig.savefig('random-forest.png', format='png', dpi=300, transparent=True)