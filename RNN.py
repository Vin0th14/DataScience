import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import models, layers
from keras.src.datasets import imdb
from keras.src.utils import pad_sequences
from sklearn.metrics import accuracy_score

max_features=10000
max_len=500
batch_size=32

print("Loading data......")
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)
print("Train sequence: ",len(x_train))
print("Test sequence: ",len(x_test))
print('Pad Sequence')
x_train=pad_sequences(x_train,maxlen=max_len)
x_test=pad_sequences(x_test,maxlen=max_len)
print("Shape of Xtrain: ",x_train.shape)
print("Shape of Xtest: ",x_test.shape)

model=models.Sequential()
model.add(layers.Embedding(max_features,32))
model.add(layers.SimpleRNN(32))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

y_pred=model.fit(x_train,y_train,epochs=10,validation_split=0.2,batch_size=128)



model=models.Sequential()
model.add(layers.Embedding(max_features,32))
model.add(layers.LSTM(32))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

y_pred=model.fit(x_train,y_train,epochs=10,validation_split=0.2,batch_size=128)






