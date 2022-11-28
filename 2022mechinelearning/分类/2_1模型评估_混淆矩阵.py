from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
y_ture=[1,0,1,1,0]
y_pred=[1,0,1,0,0]
# res=precision_score(y_ture,y_pred,average=None)#查准率
# res=confusion_matrix(y_ture,y_pred)#混淆矩阵
res=classification_report(y_ture,y_pred)#分类报告
print(res)