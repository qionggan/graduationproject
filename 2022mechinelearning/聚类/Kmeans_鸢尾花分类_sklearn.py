from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()

model = KMeans(n_clusters=3).fit(iris.data)

model.labels_


