import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

data=pd.read_csv(r'E:/Data science Notes/Python/data sets/breast_cancer.csv')
print(data.head())

X=data.drop(columns=['Class','Sample code number'])
y=data['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=24)

Model=LogisticRegression()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))