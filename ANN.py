import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score

import tensorflow as tf
import keras

# importing dataset
data=pd.read_csv(r'E:/Data science Notes/Python/data sets/Churn_Modelling.csv')
print(data.head())

# removing unwanted columns and separating Target feature
data=data.iloc[:,3:]
print(data.head())

X=data.drop(columns='Exited').values
y=data['Exited'].values
print (X)

# encoding categorical variables (Label encoder - If there is  ordinal relationship e.g ['small','medium','Large'])
le=LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])

                                    # (One hot encoder - If there is  no ordinal relationship e.g [Red, green, blue] - no relation between the values)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough') 
X = np.array(ct.fit_transform(X))
print(X)

# Splitting the Train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# Scaling the values for ANN
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train)
print(X_test)

from keras import models,layers

ann=models.Sequential()
ann.add(layers.Dense(6,activation='relu'))
ann.add(layers.Dense(6,activation='relu'))
ann.add(layers.Dense(1,activation='sigmoid'))

ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])  

ann.fit(X_train,y_train,batch_size=32,epochs=100)

y_pred=ann.predict(X_test)
y_pred=(y_pred>0.5)   # 1 if y_pred>0.5 or else 0

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))



# Parameters:

''' 
optimizer:
        > adam[Adaptive Moment Estimation] : Default choice for many deep learning problems. 
        > sgd : Works well for simple problems or when computational efficiency is critical
        > Adagrad : Sparse data problems (e.g., text or recommendation systems).
        > RMSProp : Works well for recurrent neural networks (RNNs) and sequence tasks.
        > Adadelta : When dealing with sparse data or dynamic learning rates.
        > Adamax : Large datasets or large-scale deep learning.
        > Nadam : Complex deep learning models requiring fast convergence.

    RMSprop, Adadelta, Adam have similar effects in many cases.
    adam - Default one for many developer

loss:
        > binary_crossentropy - Used for binary classification
        > categorical_crossentropy - Multi-class classification (one-hot encoded). [[0,0,1],[0,1,0],[1,0,0]]
        > sparse_categorical_crossentropy - Multi-class classification (integer labels). [0,1,2,3,4]

Activation:
        > linear : Used in the output layer for regression tasks.
        > relu : Default activation for hidden layers in deep networks due to its simplicity and efficiency.
        > softplus : Smother alternative for ReLU
        > softmax : Multi-class classification tasks (outputs probabilities for each class).
        > tanh : Works well in hidden layers for tasks where outputs need to center around zero.
        > sigmoid : Binary classification tasks (output probabilities).
        > LeakyReLU : To avoid the "dying ReLU" problem where neurons output zero for all inputs.
    
    Use cases:
    > Regression : Hidden Layer - 'relu' / '*'
                   Output Layer - 'linear' 
    > Binary classification : Hidden Layer -'relu' / '*'
                              Output Layer - 'sigmoid' 
    > Multi class classification : Hidden Layer - 'relu' / '*'
                                   Output Layer -  'softmax'

'''

