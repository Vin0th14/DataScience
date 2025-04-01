import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

data=pd.read_csv(r'E:/Data science Notes/Python/data sets/Mall_Customers.csv')
print(data.head())
print(data.info())
print(data.isnull().sum())

X = data.drop(columns=['CustomerID','Genre','Age'],axis=1).values

dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.xlabel('Cusstomers')
plt.ylabel('Euclidean distance')
plt.title('Dendrogram')
plt.show()



f_model=AgglomerativeClustering(n_clusters=5,metric='euclidean',linkage='ward')
y_pred=f_model.fit_predict(X)

print(y_pred)

plt.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100,c='red')
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100,c='blue')
plt.scatter(X[y_pred==2,0],X[y_pred==2,1],s=100,c='green')
plt.scatter(X[y_pred==3,0],X[y_pred==3,1],s=100,c='yellow')
plt.scatter(X[y_pred==4,0],X[y_pred==4,1],s=100,c='cyan')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.title('Clustering')
plt.show()