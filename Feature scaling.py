import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, Binarizer

data=pd.read_csv(r'E:/Data science Notes/Python/data sets/wine.csv',usecols=['fixed acidity','residual sugar'])
print(data.head())

# 1) Min max scalar 

MM_Sc=MinMaxScaler()
scaled_dataMM=MM_Sc.fit_transform(data)
print(scaled_dataMM)

# 2) Standard scalar
S_sc=StandardScaler()
scaled_dataSS=S_sc.fit_transform(data)
print(scaled_dataSS)

#3) Binalizer
B=Binarizer(threshold=0.5)
bin_data=B.fit_transform(data.dropna())
#print(bin_data)

# 4) get dummies (trasform catogorical into numbers 0,1,2,3) [One hot encoder]
pd.get_dummies(data,columns=['residual sugar'],drop_first=True)  # drop first will drop the first value and create new columns for other categorical value

# 5) Label encolder
le=LabelEncoder
le.fit_transform(data['residual sugar'])






