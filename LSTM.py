import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import models, layers

dataset=pd.read_csv(r'E:/Data science Notes/Python/data sets/dataset - RNN/Google_Stock_Price_Train.csv')
 

train_set=dataset.iloc[:,1:2].values    # .values - will change it to numpy array

scalar=MinMaxScaler()
train_set_scaled=scalar.fit_transform(train_set)

# creating data with 60 timesteps  and 1 output - 20 wrkng days/mnth - 60 timesteps = 3month data
# This results with RNN learn with first 60 dates and predict 61 st date and it continues till last
X_train=[]
y_train=[]
for i in range(60,1258):
    X_train.append(train_set_scaled[i-60:i,0])
    y_train.append(train_set_scaled[i,0])

X_train,y_train=np.array(X_train),np.array(y_train)

print(X_train)

# Reshaping array to 3 Dimension
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))      # We are converting 2D array to 3D array which is recomended input method for RNN()

print(X_train)

regressor=models.Sequential()

# 1st LSTM layer
regressor.add(layers.LSTM(50,return_sequences=True,input_shape=(X_train.shape[1],1))) # we will be having sequence of LSTM layers so return sequence = True
regressor.add(layers.Dropout(0.2)) #20% of neurons will be ingnored during training - Droupout regularaization to avoid overfitting

regressor.add(layers.LSTM(50,return_sequences=True)) 
regressor.add(layers.Dropout(0.2)) #20% of neurons will be ingnored during training - Droupout regularaization to avoid overfitting

regressor.add(layers.LSTM(50,return_sequences=True)) 
regressor.add(layers.Dropout(0.2)) #20% of neurons will be ingnored during training - Droupout regularaization to avoid overfitting

regressor.add(layers.LSTM(50)) 
regressor.add(layers.Dropout(0.2)) #20% of neurons will be ingnored during training - Droupout regularaization to avoid overfitting

regressor.add(layers.Dense(1))

regressor.compile(optimizer='adam',loss='mean_squared_error')

regressor.fit(X_train,y_train,epochs=100, batch_size=32)

dataset_test=pd.read_csv(r'E:/Data science Notes/Python/data sets/dataset - RNN/Google_Stock_Price_Test.csv')
Real_stock_price=dataset.iloc[:,1:2].values 

dataset_total=pd.concat((dataset['Open'],dataset_test['Open']),axis=0)
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=scalar.transform(inputs)

X_test=[]
for i in range(60,80):
    X_test.append(train_set_scaled[i-60:i,0])
X_test=np.array(X_test)


X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1)) 

predicted_stock_price=regressor.predict(X_test)
predicted_stock_price=scalar.inverse_transform(predicted_stock_price)

plt.plot(Real_stock_price,color='red',label='real price')
plt.plot(predicted_stock_price,color='Blue',label='predicted price')

plt.xlabel('time')
plt.ylabel('Google stock price')
plt.legend()
plt.show()