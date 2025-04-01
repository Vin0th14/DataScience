import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import keras
import numpy as np

from keras.src.legacy.preprocessing.image import ImageDataGenerator

from keras import models


# Data Preprocessing
# Image augmentation
traindatagen=ImageDataGenerator(                    # applying transformations in Training dataset to avoid Overfittting of data
    rescale=1./255,                 # it is used to scale the pixel values of images by a given factor
    shear_range=0.2,                
    zoom_range=0.2,
    horizontal_flip=True
)                                                      # Geomentrical transformations like rotating the image, zooming the image, shift some pixels, zoomout
train_set=traindatagen.flow_from_directory('D:/Data science Notes/Python/data sets/dataset - CNN/training_set',target_size=(64,64),batch_size=32,class_mode='binary')


testdatagen=ImageDataGenerator(rescale=1./255) # Scaling test dataset
test_set=testdatagen.flow_from_directory('D:/Data science Notes/Python/data sets/dataset - CNN/test_set',
                                               target_size=(64,64), # Image size that will be fed into the model
                                               batch_size=32,
                                               class_mode='binary')


# Building CNN - layers
cnn=keras.models.Sequential()

cnn.add(keras.layers.Conv2D(filters=32,activation='relu',kernel_size=3,input_shape=[64,64,3]))
cnn.add(keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(keras.layers.Conv2D(filters=32,activation='relu',kernel_size=3))
cnn.add(keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(keras.layers.Flatten())

cnn.add(keras.layers.Dense(units=128,activation='relu'))
cnn.add(keras.layers.Dense(units=1,activation='sigmoid'))

# Training the CNN
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])  

cnn.fit(x=train_set,validation_data=test_set,epochs=25)

# Predicting the image
 
test_image=keras.preprocessing.image.load_img('D:\Data science Notes\Python\data sets\dataset - CNN\single_prediction\cat_or_dog_1.jpg',target_size=(64,64))
test_image =keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result=cnn.predict(test_image)
print(train_set.class_indices)
print(result)
if result[0][0] == 1:
    print("Dog")
else:
    print("Cat")

 