import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel


data=pd.read_csv(r'E:/Data science Notes/Python/data sets/Housing - LR1.csv')   
print(data.info())
print(data.describe())
print(data.head())
print(data.nunique())

LE=LabelEncoder()
columnstole=['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']
data[columnstole]=data[columnstole].apply(LE.fit_transform)


X=data.drop('price',axis=1)
y=data['price']
print(X.head())
print(y.head())


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=24,test_size=0.2)

print(X_train.head())
print(y_train.head())
print(X_test.head())
print(y_test.head())

scalar=StandardScaler()
sc_X_train=scalar.fit_transform(X_train)
sc_X_test=scalar.transform(X_test)

Model=DecisionTreeRegressor(random_state=24)
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)

#print(Model.coef_)
#print(Model.intercept_)

print(r2_score(y_test,y_pred))
print(root_mean_squared_error(y_test,y_pred))

