import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

data=pd.read_csv(r'E:/Data science Notes/Python/data sets/Mall_Customers.csv')
print(data.head())
print(data.info())
print(data.isnull().sum())

data = data.drop(columns=['CustomerID','Genre','Age'],axis=1).values

wcss=[]
for i in range(1,11):
    model=KMeans(n_clusters=i,init='k-means++',random_state=42)
    model.fit(data)
    wcss.append(model.inertia_)

'''plt.plot(range(1,11),wcss)
plt.title("Elbow chart")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()'''   # from the diagram we get to know the optiomal no of clusters is 5

f_model=KMeans(n_clusters=5,init="k-means++",random_state=42)
y_pred=f_model.fit_predict(data)

print(y_pred)

plt.scatter(data[y_pred==0,0],data[y_pred==0,1],s=100,c='red')
plt.scatter(data[y_pred==1,0],data[y_pred==1,1],s=100,c='blue')
plt.scatter(data[y_pred==2,0],data[y_pred==2,1],s=100,c='green')
plt.scatter(data[y_pred==3,0],data[y_pred==3,1],s=100,c='yellow')
plt.scatter(data[y_pred==4,0],data[y_pred==4,1],s=100,c='cyan')
plt.scatter(f_model.cluster_centers_[:,0],f_model.cluster_centers_[:,1],s=200,c='black')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.title('Clustering')
plt.show()