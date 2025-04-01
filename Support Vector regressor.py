import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,root_mean_squared_error

data=pd.read_csv('E:/Data science Notes/Python/data sets/Position_Salaries.csv')
print(data.head())

X=data.drop(columns=['Salary','Position'],axis=1)
y=data['Salary']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=24,test_size=0.2)



scalar=StandardScaler()
sc_X_train=scalar.fit_transform(X_train)
sc_X_test=scalar.transform(X_test)

Model=SVR(kernel='rbf')
Model.fit(X,y)
y_pred=Model.predict(sc_X_test)


#print(Model.coef_)
#print(Model.intercept_)

print(r2_score(y_test,y_pred))
print(root_mean_squared_error(y_test,y_pred))