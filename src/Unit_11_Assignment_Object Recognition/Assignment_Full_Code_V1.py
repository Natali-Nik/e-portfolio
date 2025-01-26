#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 20:13:51 2025

@author: natalinikolic
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential       #to define model/ layers
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten


# load and reformat the training data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


meta = unpickle('batches.meta')

datafiles = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

datadic = dict()

for file in datafiles:
    tmp = unpickle(file)
    datadic.update(tmp)


# load the test data
testdat = unpickle('test_batch')


x = datadic[b'data'] 
y = datadic[b'labels']
xtest = testdat[b'data'] 
ytest = testdat[b'labels']

# Randomly split xtest into two equal-sized datasets: xtest_1 and xtest_2
from sklearn.model_selection import train_test_split

# Assuming xtest and ytest (labels) are already defined in the CNN code
xtest_1, xtest_2, ytest_1, ytest_2 = train_test_split(xtest, ytest_cat, test_size=0.5, random_state=42)

# Display the sizes of the resulting splits
print("Size of xtest_1:", len(xtest_1))
print("Size of xtest_2:", len(xtest_2))

# Optionally inspect the first few rows of the splits
print("First few samples of xtest_1:")
print(xtest_1[:5])
print("First few labels of ytest_1:")
print(ytest_1[:5])


# preprocess the data  prior to training
x_scale = x/x.max()
xtest_scale = xtest/x.max()
y_cat = to_categorical(y,10)
ytest_cat = to_categorical(ytest, 10)


model = Sequential()

## ************* FIRST SET OF LAYERS *************************

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

## *************** SECOND SET OF LAYERS ***********************
#Since the shape of the data is 32 x 32 x 3 =3072 ...
#We need to deal with this more complex structure by adding yet another convolutional layer

# *************CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 32 x 32 x 3 =3072 BEFORE FINAL LAYER
model.add(Flatten())

# 256 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(256, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



history = model.fit(x_scale,y_cat,epochs=25)
                    
                    
                    