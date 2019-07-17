# Vinayak Gajjewar 7/16/19

from __future__ import absolute_import, division, print_function, unicode_literals

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras

# Import helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Print TensorFlow's version number to affirm that it's installed correctly
#print(tf.__version__)

# Import the Fashion MNIST dataset
# Slightly more challenging of a problem for the neural net than vanilla MNIST
# Loading the data returns 4 28x28 NumPy arrays with pixel values ranging from 0 to 255
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# The names of the different labels (0-9)
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Print the shape of the training images array
#print("Shape of the training images:")
#print(train_images.shape)

# Print the length of the labels array
#print("Length of the training labels array:")
#print(len(train_labels))

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
# Uncomment this if you want to see the first 25 images
# plt.show()

# Configure the layers of the NN
# The first layer flattens the data from a 28x28 array to a 1x784 array
# The hidden layer has 128 neurons and uses the Rectified Linear Unit activation function
# ReLU uses the function f(x)=max(0, x)
# The output layer has 10 neurons and uses the softmax activation function
# It returns an array of 10 probability scores (1 for each label) that sum up to 1
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model and specify which optimizer and loss function to use
# The optimizer (which updates the gradients across the NN) is called Stochastic Gradient Descent
# Don't use mean squared error (MSE) for the loss function because it expects the input to be in the same shape as the output
# Which is not possible
# Use sparse categorical crossentropy instead and train for accuracy
model.compile(
    optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Iterate over the training data 5 times
# 10 epochs is overkill
model.fit(train_images, train_labels, epochs=5)

# Evaluate the accuracy of the model using the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy: ", test_acc)

# Predict the classifications for the test data set
# A single prediction is an array of 10 numbers that contains the confidence of the NN that the image is one of the 10 labels
# np.argmax() returns the largest prediction value (whichever label the NN is most confident in)
predictions = model.predict(test_images)

for i in range(25):
    print(str(i) + ": " + class_names[np.argmax(predictions[i])] + " confidence = " + str(predictions[i][np.argmax(predictions[i])]))

print(model.summary())