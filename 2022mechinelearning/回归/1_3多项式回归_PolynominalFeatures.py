from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
X = np.arange(6).reshape(3, 2)
X
poly = PolynomialFeatures(degree=2)
# poly = PolynomialFeatures(degree=2,interaction_only=True)
poly.fit_transform(X)
poly.get_feature_names()
# # 转换成DataFrame
# poly2 = PolynomialFeatures(degree=3)
# X_2 = poly2.fit_transform(X)
# df=pd.DataFrame(data = X_2 ,columns=poly2.get_feature_names())
