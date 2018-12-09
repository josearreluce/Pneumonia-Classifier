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




from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten


class model():
    def __init__(self, batch_size=100, epochs=100, verbose=1):
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.model = self.create_model()

        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def create_model():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nClasses, activation='softmax'))

        return model

    def train(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = np_utils.to_categorical(y_train, num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes)

        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(x_train)

        # here's a more "manual" example
        for e in range(epochs):
            print('Epoch', e)
            batches = 0
            for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
                model.fit(x_batch, y_batch) # TODO NEEDS CALLBACKS
                batches += 1
                if batches >= len(x_train) / 32:
                    break

    def evaluate(self):
        self.model.evaluate(test_data, test_labels_one_hot)



