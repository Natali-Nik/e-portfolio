#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 20:46:26 2025

@author: natalinikolic
"""

#Based on copied code from CNN example. Aim is to adjust to new data from CIFAR10




## Dataset Description
#CIFAR-10 is an established computer-vision dataset used for object recognition. It is a subset of the 80 million tiny images dataset and consists of 60,000 32x32 color images containing one of 10 object classes, with 6000 images per class. It was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.

#Import libraries
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.datasets import cifar10


# Set random seeds for reproducibility
seed(888)
tf.random.set_seed(112)

# Load the CIFAR-10 dataset
(x_train_all, y_train_all), (x_test, y_test) = cifar10.load_data()

# Define label names
LABEL_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

## Data Exploration
# Normalize data to the range [0, 1]
x_train_all = x_train_all / 255.0
x_test = x_test / 255.0

# Display dataset details
print(f"Training data shape: {x_train_all.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Number of classes: {len(LABEL_NAMES)}")

# Visualize a sample image
plt.imshow(x_train_all[0])
plt.title(f"Class: {LABEL_NAMES[y_train_all[0][0]]}")
plt.show()

# Combine CIFAR-10 training and test sets
x_all = np.concatenate([x_train_all, x_test], axis=0)
y_all = np.concatenate([y_train_all, y_test], axis=0)

# Perform an 80/20 split (training + temp)
x_train, x_temp, y_train, y_temp = train_test_split(
    x_all, y_all, test_size=0.2, random_state=42
)

# Split the temp set into 10% validation and 10% test
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42
)

# One-hot encode the labels after splitting
y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


# Confirm the overall sizes of the splits
total_samples = x_all.shape[0]
print(f"Training set size: {x_train.shape[0]}, y_train shape: {y_train.shape}")
print(f"Validation set size: {x_val.shape[0]}, y_val shape: {y_val.shape}")
print(f"Test set size: {x_test.shape[0]}, y_test shape: {y_test.shape}")

# Check label distribution in the dataset
unique, counts = np.unique(y_train_all, return_counts=True)
print("Label distribution in the training set:")
for label, count in zip(LABEL_NAMES, counts):
    print(f"{label}: {count}")

# Visualize a grid of sample images
fig, axes = plt.subplots(3, 5, figsize=(10, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train_all[i])
    ax.set_title(LABEL_NAMES[y_train_all[i][0]])
    ax.axis('off')
plt.tight_layout()
plt.show()

# Utility function to visualize a grid of images
def visualize_samples(images, labels, label_names, grid=(3, 5)):
    fig, axes = plt.subplots(grid[0], grid[1], figsize=(10, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(label_names[np.argmax(labels[i])])  # Use np.argmax for one-hot labels
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Visualize sample images from the training set
visualize_samples(x_train, y_train, LABEL_NAMES)


##Model Architecture

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Early stopping to prevent overfitting
from tensorflow.keras.callbacks import EarlyStopping

# Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])  # Output layer with 10 classes

# Compile the Model
model.compile(
    optimizer=Adam(learning_rate=0.0004),  # Reduced learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss',
                           patience=4,  # Increased patience
                           restore_best_weights=True,
                           verbose=1)  # Verbose logging

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.15
)
datagen.fit(x_train)

# Train the Model
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=35,
    validation_data=(x_val, y_val),
    callbacks=[early_stop]
)

# Plot Training and Validation Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

# Plot Training and Validation Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Generate predictions
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Generate classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))

# Classification Report and Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Confirm sizes for debugging
print(f"x_test shape: {x_test.shape}")  # Should be (6000, 32, 32, 3)
print(f"y_test shape: {y_test.shape}")  # Should match model's expectations


# Convert one-hot encoded labels to integers (if needed)
y_test_labels = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test.flatten().astype(int)

# Predict and visualize a single image
index = 12
predicted_label = LABEL_NAMES[y_pred[index]]
true_label = LABEL_NAMES[y_true[index]]

plt.imshow(x_test[index])
plt.title(f"True: {true_label}, Predicted: {predicted_label}")
plt.axis('off')
plt.show()



