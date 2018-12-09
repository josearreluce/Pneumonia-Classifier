import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy.misc import imread
from skimage.transform import resize

from os import listdir
"""
# Gather training data
pneumonia_training_directory = './data/train/PNEUMONIA/'
normal_training_directory = './data/train/NORMAL/'

training_images, training_labels = [], []

pcount = 0
for filename in listdir(pneumonia_training_directory):
    image = imread(pneumonia_training_directory + filename)
    image = resize(image, (224, 224))
    image = image.tolist()
    image = np.array(image)

    training_images.append(image)
    training_labels.append(1)
    pcount += 1
    #if pcount > 100:
    #    break

ncount = 0
for filename in listdir(normal_training_directory):
    image = imread(normal_training_directory + filename)
    image = resize(image, (224, 224))
    image = image.tolist()
    image = np.array(image)

    training_images.append(image)
    training_labels.append(0)
    ncount += 1
    #if ncount > 100:
    #    break

test_images = []
test_labels = []

normal_test_directory = './data/test/NORMAL/'
pneumonia_test_directory = './data/test/PNEUMONIA/'

count = 0
for filename in listdir(normal_test_directory):
    image = imread(normal_test_directory + filename)
    image = resize(image, (224, 224))

    test_images.append(image)
    test_labels.append(0)
    count += 1
    #if count > 100:
    #    break
count = 0
for filename in listdir(pneumonia_test_directory):
    image = imread(pneumonia_test_directory + filename)
    image = resize(image, (224, 224))

    test_images.append(image)
    test_labels.append(1)
    count += 1
    #if count > 100:
    #    break
# Convert images to grayscale if not already in grayscale
for i in range(len(training_images)):
    if training_images[i].ndim > 2:
        training_images[i] = np.dot(training_images[i][..., :3], [0.29894, 0.58704, 0.11402])
training_images = np.array(training_images)
training_images = training_images / 255.0

training_labels = np.array(training_labels)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(224, 224)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50)

# Gather the Test images too
test_images = np.array(test_images)
test_labels = np.array(test_labels)

test_images = test_images / 255.0

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Accuracy: ", test_acc)
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class model():
    def __init__(self, batch_size=100, epochs=100, verbose=1):
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.model = self.create_model()
        self.numClasses = 2
        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.test_data = []
        self.test_labels = []

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 5218)))
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
        model.add(Dense(self.numClasses, activation='softmax'))

        return model

    def train(self):
        #(x_train, y_train), (x_test, y_test) = self.load_data()
        train_images, train_labels = self.load_training_data()
        test_images, test_labels = self.load_testing_data()

        #y_train = np_utils.to_categorical(y_train, num_classes)
        #y_test = np_utils.to_categorical(y_test, num_classes)

        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        #datagen.fit(x_train)
        datagen.fit(train_images)

        # here's a more "manual" example
        epochs = 5
        for e in range(epochs):
            print('Epoch', e)
            batches = 0
            for x_batch, y_batch in datagen.flow(train_images, train_labels, batch_size=32):
                model.fit(x_batch, y_batch) # TODO NEEDS CALLBACKS
                batches += 1
                if batches >= len(train_images) / 32:
                    break

    def evaluate(self):
        self.model.evaluate(self.test_data, self.test_labels)

    def load_training_data(self):
        pneumonia_training_directory = './data/train/PNEUMONIA/'
        normal_training_directory = './data/train/NORMAL/'

        training_images, training_labels = [], []

        for filename in listdir(pneumonia_training_directory):
            image = imread(pneumonia_training_directory + filename)
            image = resize(image, (224, 224))
            image = image.tolist()
            image = np.array(image)

            training_images.append(image)
            training_labels.append(1)

        for filename in listdir(normal_training_directory):
            image = imread(normal_training_directory + filename)
            image = resize(image, (224, 224))
            image = image.tolist()
            image = np.array(image)

            training_images.append(image)
            training_labels.append(0)

        return training_images, training_labels

    def load_testing_data(self):
        test_images = []
        test_labels = []

        normal_test_directory = './data/test/NORMAL/'
        pneumonia_test_directory = './data/test/PNEUMONIA/'

        for filename in listdir(normal_test_directory):
            image = imread(normal_test_directory + filename)
            image = resize(image, (224, 224))

            test_images.append(image)
            test_labels.append(0)

        for filename in listdir(pneumonia_test_directory):
            image = imread(pneumonia_test_directory + filename)
            image = resize(image, (224, 224))

            test_images.append(image)
            test_labels.append(1)

        self.test_data = test_images
        self.test_labels = test_labels

        return test_images, test_labels

new_model = model()
new_model.train()
print(new_model.evaluate())