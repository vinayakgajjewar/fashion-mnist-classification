# Vinayak Gajjewar 7/16/19

from __future__ import absolute_import, division, print_function, unicode_literals

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras

# Import helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Print TensorFlow's version number to affirm that it's installed correctly
print(tf.__version__)

# Import the Fashion MNIST dataset
# Slightly more challenging of a problem for the neural net than vanilla MNIST
# Loading the data returns 4 28x28 NumPy arrays with pizel values ranging from 0 to 255
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# The names of the different labels (0-9)
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Print the shape of the training images array
print("Shape of the training images:")
print(train_images.shape)

# Print the length of the labels array
print("Length of the training labels array:")
print(len(train_labels))

# Show the first training image
#plt.figure()
#plt.imshow(train_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()

# Scale the images to values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Display the first 25 images from the training set along with their labels
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()