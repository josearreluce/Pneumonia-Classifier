import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy.misc import imread
from skimage.transform import resize
from os import listdir
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json

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

class model():
    def __init__(self, batch_size=100, epochs=50, verbose=1):
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.numClasses = 2
        self.test_data = []
        self.test_labels = []

        self.model = self.create_model()
        self.model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(5528, 224, 224, 1)))
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
        print('Training Data Collected')
        #test_images, test_labels = self.load_testing_data()

        #y_train = np_utils.to_categorical(y_train, num_classes)
        #y_test = np_utils.to_categorical(y_test, num_classes)

        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            horizontal_flip=True)
        print('Datagen')
        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        # datagen.fit(x_train)
        datagen.fit(train_images)
        print('Datagen fitted')

        """
        self.model.fit_generator(
                datagen.flow(train_images, train_labels, batch_size=self.batch_size),
                steps_per_epoch=len(train_images) / self.batch_size,
                epochs=self.epochs,
                verbose=2)
        print('model fitted')
        """
        # here's a more "manual" example
        for e in range(self.epochs):
            print('Epoch:', e)
            batches = 0
            #for x_batch, y_batch in datagen.flow(train_images, train_labels, batch_size=self.batch_size):
            print('     Batch:', batches)
            #model.fit(x_batch, y_batch, verbose=1) # TODO NEEDS CALLBACKS
            self.model.fit(train_images, train_labels, verbose=1) # TODO NEEDS CALLBACKS
            batches += 1
            if batches >= len(train_images) / self.batch_size:
                break

    def evaluate(self):
        self.model.evaluate(self.test_data, self.test_labels)

    # from hw2 assignment
    def rgb2gray(self, rgb):
        gray = np.dot(rgb[...,:3],[0.29894, 0.58704, 0.11402])
        return gray

    def unison_shuffled_copies(self, images, labels):
        assert len(images) == len(labels)
        p = numpy.random.permutation(len(images))
        return images[p], labels[p]

    def _load_data(self, directory, labels):
        images, labels = [], []

        for filename in listdir(directory):
            try:
                image = imread(directory + filename)
                image = image / 255
                if (image.ndim == 3): # TODO Confirm if necessary
                    print(filename, image.shape)
                    image = self.rgb2gray(image)
                image = resize(image, (224, 224, 1))
                images.append(image)
                labels.append(labels)
            except Exception as i:
                print('CAUGHT: ', i)
        return images, labels

    def load_training_data(self):
        pneumonia_training_directory = './data/train/PNEUMONIA/'
        normal_training_directory = './data/train/NORMAL/'

        training_images, training_labels = self._load_data(pneumonia_training_directory, 1)
        training_images_2, training_labels_2 = self._load_data(normal_training_directory, 0)
        training_images = np.concatenate((training_images, training_images_2), axis=0)
        training_labels.extend(training_labels_2)

        print('Training Data Details:', training_images.shape, len(training_labels))
        return training_images, training_labels

    def load_testing_data(self):
        normal_test_directory = './data/test/NORMAL/'
        pneumonia_test_directory = './data/test/PNEUMONIA/'

        test_images, test_labels = self._load_data(pneumonia_test_directory, 1)
        test_images_2, test_labels_2 = self._load_data(normal_test_directory, 0)

        test_images = np.concatenate((test_images, test_images_2), axis=0)
        test_labels = np.concatenate((test_labels, test_labels_2), axis=0)

        self.test_data = test_images
        self.test_labels = test_labels

        print('Test Data Details:', test_images.shape, test_labels.shape)
        return test_images, test_labels

    # serialize model to JSON
    def save_model(self):
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")

    # load json and create model
    def load_model(self):
        with open('model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        self.model = loaded_model
        print("Loaded model from disk")

new_model = model(epochs=1, batch_size=100)
new_model.train()
print(new_model.evaluate())
new_model.save_model()
