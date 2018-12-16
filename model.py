import numpy as np
from os import listdir
from sklearn.utils import shuffle
from scipy.misc import imread
from skimage.transform import resize
from tensorflow.train import AdamOptimizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
from visualize import plot_training_alt, plot_training


class model():
    def __init__(self, batch_size=100, epochs=50, verbose=1):
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.num_classes = 2
        self.test_data = None
        self.test_labels = None
        self.tracker = None
        self.img_shape = (224, 224, 1)

        self.model = self.create_model()
        self.model.compile(optimizer=AdamOptimizer(),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def create_model(self):
        """
        Creates our best (so far) sequential CNN and prepares it for training.
        """
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(244, 244, 1)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        print(model.summary())
        return model

    def augment_data(self):
        """
        Creates a data augmentation generator to be used for
        random augmentation of training batch data

        :return: the generator to be trained and used
        """
        generator = ImageDataGenerator(
            featurewise_std_normalization=True,
            horizontal_flip=True,
            data_format='channels_last')

        return generator

    def train(self):
        """
        Collects training and testing data, starts data augmentation generators for each, and
        fits the model with the test data, validating with the testing data after each epoch.

        Once done, each epochs loss, acc, val_loss, and val_acc will be printed, and those
        values will be greated into multiple plots and saved for viewing.
        """
        train_images, train_labels, label_weights  = self.load_training_data()
        test_images, test_labels = self.load_testing_data()
        print('Training Data Collected')
        training_gen = self.augment_data()
        training_gen.fit(train_images)

        test_gen = ImageDataGenerator(
            featurewise_std_normalization=True,
            horizontal_flip=True,
            data_format='channels_last')
        test_gen.fit(test_images)

        tracker = self.model.fit_generator(
                training_gen.flow(train_images, train_labels, batch_size=self.batch_size),
                epochs=self.epochs,
                steps_per_epoch=len(train_images) // self.batch_size,
                class_weight=label_weights,
                shuffle=True,
                validation_data=test_gen.flow(test_images, test_labels, batch_size=self.batch_size),
                validation_steps=len(test_images) // self.batch_size,
                verbose=self.verbose)

        print('ACC',tracker.history['acc'])
        print('LOSS', tracker.history['loss'])
        print('VAL ACC', tracker.history['val_acc'])
        print('VAL LOSS', tracker.history['val_loss'])
        plot_training(tracker)
        plot_training_alt(tracker, self.epochs)

    def evaluate(self):
        """
        Evaluates the model over the entire testing dataset to get an overall accuracy and loss.
        """
        if len(self.test_data) == 0:
            print('Getting Test Data')
            self.load_testing_data()

        evaluate = self.model.evaluate(
                self.test_data,
                self.test_labels,
                verbose=1)
        print(evaluate) # [loss, accuracy]
        return evaluate

    def rgb2gray(self, rgb):
        """
        FROM HW2 Assignment.
        Condenses RGB image into 1 channel grayscale.
        """
        gray = np.dot(rgb[...,:3],[0.29894, 0.58704, 0.11402])
        return gray

    def _load_data(self, directory, label):
        """
        Given a directory and label, loads every image of that directory,
        as a numpy array, into a set and assigns the label to each item in
        that set accordingly. Will condense RGB images into 1 channel.

        :directory: directory filepath to look for images
        :label: label to assign to each image

        :return: a list of images as numpy arrays and their associated labels
        """
        images, labels = [], []
        files = listdir(directory)
        for filename in files:
            try:
                image = imread(directory + filename)
                image = image / 255
                if image.ndim == 3:
                    image = self.rgb2gray(image)
                image = resize(image, self.img_shape)
                images.append(image)
                labels.append(label)
            except Exception as i:
                print('CAUGHT: ', i)

        return np.array(images), np.array(labels)

    def load_training_data(self):
        """
        Loads training pneumonia and normal images into a two lists, randomizes them, and
        generates a class weight dict for training purposes.

        :return: shuffled array of image data, shuffled labels (in the same order), and wieghts dict
        """
        pneumonia_training_directory = './data/train/PNEUMONIA/'
        normal_training_directory = './data/train/NORMAL/'

        training_images_sick, training_labels_sick = self._load_data(pneumonia_training_directory, 1)
        training_images_norm, training_labels_norm = self._load_data(normal_training_directory, 0)

        training_images = np.concatenate((training_images_sick, training_images_norm), axis=0)
        training_labels = np.concatenate((training_labels_sick, training_labels_norm), axis=0)

        shuff_images, shuff_labels = shuffle(training_images, training_labels, random_state=5)

        print('Training Data Details- Images:', shuff_images.shape, 'Labels:', shuff_labels.shape)
        # See visualize.py for logic
        weight_dict = {
                1 : 1,
                0 : len(training_images_sick) // len(training_images_norm)
            }
        return shuff_images, shuff_labels, weight_dict

    def load_testing_data(self):
        """
        Loads test pneumonia and normal images into a two lists and randomizes them

        :return: test images in an array and thier labels
        """
        normal_test_directory = './data/test/NORMAL/'
        pneumonia_test_directory = './data/test/PNEUMONIA/'

        test_images_sick, test_labels_sick = self._load_data(pneumonia_test_directory, 1)
        test_images_norm, test_labels_norm = self._load_data(normal_test_directory, 0)

        test_images = np.concatenate((test_images_sick, test_images_norm), axis=0)
        test_labels = np.concatenate((test_labels_sick, test_labels_norm), axis=0)

        self.test_data = test_images
        self.test_labels = test_labels

        print('Test Data Details- Images:', test_images.shape, 'Labels:', test_labels.shape)
        return test_images, test_labels

    def save_model(self, filename='model'):
        """
        Save the model structure as filename.json and the trained model weights
        as filename.h5

        :filename: the name of the file to save the model as
        """
        model_json = self.model.to_json()
        with open(filename + '.json', 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights(filename + '.h5')
        print('Saved model to as', filename)

    def load_model(self, filename='model'):
        """
        load the model structure and model weights (must be named the same) from
        filename.json and filename.h5 files

        :filename: the model filename to load
        """
        with open(filename + '.json', 'r') as json_file:
            loaded_model_json = json_file.read()

        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(filename + '.h5')

        self.model = loaded_model
        print('Loaded model from', filename)

new_model = model(epochs=4, batch_size=100)
new_model.train()
print(new_model.evaluate())
new_model.save_model(filename='model')
