import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,root_mean_squared_error

data=pd.read_csv('E:/Data science Notes/Python/data sets/Position_Salaries.csv')
print(data.head())

X=data.drop(columns=['Salary','Position'],axis=1)
y=data['Salary']

PF=PolynomialFeatures(degree=4)
p_X=PF.fit_transform(X)

PX_train,PX_test,y_train,y_test=train_test_split(p_X,y,random_state=24,test_size=0.2)

Model=LinearRegression()                               # Polynomial regression
Model.fit(PX_train,y_train)
y_pred=Model.predict(PX_test)

print(Model.coef_)
print(Model.intercept_)

print(r2_score(y_test,y_pred))
print(root_mean_squared_error(y_test,y_pred))

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=24,test_size=0.2)

Model=LinearRegression()                              # Lineaar regression
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)

print(Model.coef_)
print(Model.intercept_)

print(r2_score(y_test,y_pred))
print(root_mean_squared_error(y_test,y_pred))

# Linear regression performs very poorly and Polynomial performs very good level
