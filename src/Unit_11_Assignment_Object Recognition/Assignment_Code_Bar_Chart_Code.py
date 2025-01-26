#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:54:54 2025

@author: natalinikolic
"""


from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the CIFAR-10 dataset
(_, y_train), (_, _) = cifar10.load_data()

# Get class distribution
class_counts = np.bincount(y_train.flatten(), minlength=10)
class_labels = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Plot the class distribution
plt.figure(figsize=(10, 6))
plt.bar(class_labels, class_counts, color="skyblue")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Class Distribution in CIFAR-10 Dataset")
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot as an image
plt.savefig("cifar10_class_distribution.png")
plt.show()

# Export class distribution to CSV
class_distribution = pd.DataFrame({
    "Class": class_labels,
    "Count": class_counts
})
class_distribution.to_csv("cifar10_class_distribution.csv", index=False)
