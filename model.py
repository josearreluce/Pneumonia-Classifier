import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy.misc import imread
from skimage.transform import resize
from os import listdir
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
from sklearn.utils import shuffle
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt
import timeit
import os
import glob
import re
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
        self.img_shape = (224, 224, 3)

        self.model = self.create_model()
        self.model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.00001),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def create_model(self, dropout_rate=0.5):       
        vgg_model = VGG16(weights='imagenet', input_shape=self.img_shape, include_top=False)
        for layer in vgg_model.layers[:-4]:
            layer.trainable = False

        x = Sequential()
        x.add(vgg_model)
        x.add(Flatten())
        x.add(Dense(1024, activation='relu'))
        x.add(Dropout(0.5))
        x.add(Dense(512, activation = 'softmax'))
        return x
        # x = Flatten()(vgg_model.output)
        #
        # x = Dense(1024, use_bias=False)(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        # x = Dropout(dropout_rate)(x)
        #
        # x = Dense(512, use_bias=False)(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        # x = Dropout(dropout_rate)(x)

        predictions = Dense(1, activation='softmax')(x)
        
        model = Model(vgg_model.input, predictions)
        print(model.summary())
        return model
        #model = Sequential()
        '''
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=self.img_shape))
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

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu', name='final_model_conv'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu')) # TODO CONFIRM
        model.add(Dense(1, activation='sigmoid'))
        print(model.summary())
        # TODO add/refine models
        return model
        '''

    def augment_data(self):
        generator = ImageDataGenerator(
            #featurewise_center=True,
            featurewise_std_normalization=True,
            width_shift_range=0.10,
            height_shift_range=0.10,
            brightness_range=(1.0, 1.2),
            zoom_range=0.05,
            rotation_range=0.10,
            # TODO brightness_range?
            # TODO zoom_range?
            # TODO some rotation?
            # TODO add vert and horz offset ranges
            data_format='channels_last',
            horizontal_flip=True)
        return generator

    def train(self):
        train_images, train_labels, label_weights  = self.load_training_data()
        test_images, test_labels = self.load_testing_data() # TODO use val not test
        print('Training Data Collected')
        '''
        training_gen = ImageDataGenerator(
            #featurewise_center=True,
            featurewise_std_normalization=True,
            # TODO brightness_range?
            # TODO add vert and horz offset ranges
            data_format='channels_last',
            horizontal_flip=True)
        '''
        training_gen = self.augment_data()
        training_gen.fit(train_images)

        test_gen = self.augment_data()
        test_gen.fit(test_images)

        '''
        breaker = keras.callbacks.EarlyStopping( # TODO Use this to test until val_loss is good enough or not improving
                monitor='val_loss',
                min_delta=0,
                patience=0,
                verbose=0,
                mode='auto',
                baseline=None,
                restore_best_weights=False)
        '''

        tracker = self.model.fit_generator(
                training_gen.flow(train_images, train_labels, batch_size=self.batch_size),
                epochs=self.epochs,
                steps_per_epoch=len(train_images) // self.batch_size,
                class_weight=label_weights,  # TODO needs class weights
                shuffle=True, # TODO conf if necessary,
                validation_data=test_gen.flow(test_images, test_labels, batch_size=self.batch_size), # TODO this should be using images from val not test
                validation_steps=len(test_images) // self.batch_size, # TODO confirm stesp
                verbose=self.verbose)
                # TODO needs test batches for beter epoch eval

        print('Accuracy', tracker.history['acc'])
        print('Val Acc', tracker.history['val_acc'])
        print('Loss', tracker.history['loss'])
        print('Val Loss', tracker.history['val_loss'])
        plot_training(tracker)
        plot_training_alt(tracker, self.epochs)

    def evaluate(self):
        if len(self.test_data) == 0:
            print('Getting Test Data')
            self.load_testing_data()

        evaluate = self.model.evaluate(
                self.test_data,
                self.test_labels,
                #steps=len(self.test_data) // self.batch_size,
                verbose=1)
        print(evaluate) # TODO [loss, accuracy]
        return evaluate

    # from hw2 assignment
    def rgb2gray(self, rgb):
        gray = np.dot(rgb[...,:3],[0.29894, 0.58704, 0.11402])
        return gray

    def _load_data(self, directory, label):
        images, labels = [], []
        files = listdir(directory)
        count = 0
        for filename in files:
            try:
                image = imread(directory + filename)
                image = image / 255
                if image.ndim != 3:
                    image = np.array([[image], [image], [image]])
                    image = self.rgb2gray(image)
                image = resize(image, self.img_shape)
                images.append(image)
                labels.append(label)
            except Exception as i:
                print('CAUGHT: ', i)
            """
            count += 1
            if count > 80:
                break
            """

        return np.array(images), np.array(labels)

    def load_training_data(self):
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

    # serialize model to JSON
    def save_model(self):
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
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

new_model = model(epochs=50, batch_size=100)
new_model.train()
print(new_model.evaluate())
new_model.save_model()
