#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 18:38:58 2025

@author: natalinikolic
"""


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
from numpy.random import seed
seed(888)
tf.random.set_seed(112)

# Load the CIFAR-10 dataset
from keras.datasets import cifar10

(x_train_all, y_train_all), (x_test, y_test) = cifar10.load_data()

# CIFAR-10 class labels
LABEL_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Display dataset shape and details
print(f"Training data shape: {x_train_all.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Number of classes: {len(LABEL_NAMES)}")

# Example of displaying an image
plt.imshow(x_train_all[0])
plt.title(f"Class: {LABEL_NAMES[y_train_all[0][0]]}")
plt.show()

# Normalize the data to the range [0, 1]
x_train_all = x_train_all / 255.0
x_test = x_test / 255.0

# Convert labels to one-hot encoding
y_cat_train_all = to_categorical(y_train_all, 10)
y_cat_test = to_categorical(y_test, 10)

# Create a validation set (10% of training data)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_cat_train_all, test_size=0.1, random_state=42)

print(f"Training set size: {x_train.shape[0]}")
print(f"Validation set size: {x_val.shape[0]}")

# Define the CNN model
model = Sequential()

# First set of layers
model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

# Second set of layers
model.add(Conv2D(filters=64, kernel_size=(4, 4), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

# Flatten the feature maps
model.add(Flatten())

# Fully connected dense layers
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Early stopping to prevent overfitting
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(x_train, y_train, epochs=25, batch_size=64,
                    validation_data=(x_val, y_val), callbacks=[early_stop])

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training vs. Validation Loss')
plt.show()

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training vs. Validation Accuracy')
plt.show()

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_cat_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Classification report and confusion matrix
y_pred = np.argmax(model.predict(x_test), axis=-1)
y_true = np.argmax(y_cat_test, axis=-1)

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Example: Predicting a single image
index = 12
plt.imshow(x_test[index])
plt.title(f"True Label: {LABEL_NAMES[y_test[index][0]]}")
plt.show()

single_image = x_test[index].reshape(1, 32, 32, 3)
prediction = np.argmax(model.predict(single_image), axis=-1)
print(f"Predicted Label: {LABEL_NAMES[prediction[0]]}")
