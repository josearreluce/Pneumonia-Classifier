import tensorflow as tf
from tensorflow import keras
from os import listdir
from scipy.misc import imread # May be outdated see: https://stackoverflow.com/questions/25102461/python-rgb-matrix-of-an-image
import numpy as np
import pickle

# Load the training and testing data
pneumonia_training_directory = './data/train/PNEUMONIA/'
normal_training_directory = './data/train/NORMAL/'

training_images = []
training_labels = []
print("Gathering Training Images")
pcount = 0
for filename in listdir(pneumonia_training_directory):
    image = imread(pneumonia_training_directory + filename)
    #pneumonia_training_images.append((image, 1))
    training_images.append(image)
    training_labels.append(1)
    print(pcount)
    pcount += 1
print("FINISHED GATHERING PNEUMONIA")
ncount = 0
for filename in listdir(normal_training_directory):
    image = imread(normal_training_directory + filename)
    print(image.shape)
    #normal_training_images.append((image, 0))
    training_images.append(image)
    training_labels.append(0)
    print(ncount)
    ncount += 1
print("FINISHED GATHERING NORMAL")

training_images = np.array(training_images)
training_labels = np.array(training_labels)

exit()
training_images = training_images / 255.0


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

# Gather the Test images too
test_images = np.array(test_images)
test_labels = np.array(test_labels)

test_images = test_images / 255.0

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Accuracy: ", test_acc)