import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy.misc import imread
from skimage.transform import resize
from os import listdir
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tensorflow.keras.models import Model
from tensorflow.keras import backend
from skimage.transform import resize

class evaluateModel():
    def __init__(self):
        self.model = self.load_model()
        self.eval_dir = './data/val/'

    def load_model(self):
        with open('model.json', 'r') as json_file:
            loaded_model_json = json_file.read()

        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")

        loaded_model.compile(optimizer=tf.train.AdamOptimizer(),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        print(loaded_model.summary())
        return loaded_model

    def add_prob(self, image):
        prediction = self.model.predict(image)
        sick = prediction[0,0]
        norm = 1 - sick
        proba, label = (norm, "Not Infected") if norm > sick else (sick, "Pneumonia")
        label = "{}: {}%".format(label, round(proba * 100, 2))
        return label

    # from hw2 assignment
    def rgb2gray(self, rgb):
        gray = np.dot(rgb[...,:3],[0.29894, 0.58704, 0.11402])
        return gray

    def class_activation_map(self, basic_img, img):

        preds = self.model.predict(img)
        output = self.model.output[:, 0]
        last_conv_layer = self.model.get_layer('final_model_conv')
        grads = K.gradients(output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

        iterate = K.function([self.model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([img])

        for i in range(last_conv_layer.output.shape[-1]):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        cam = np.mean(conv_layer_output_value, axis=-1)
        cam = np.maximum(cam, 0)
        cam /= np.max(cam)
        mapping = cm.get_cmap('rainbow_r')
        heatmap = mapping(np.uint8(255 * cam))
        #heatmap[np.where(cam > 0.5)] = 0 # thresholding
        heatmap = resize(heatmap, (basic_img.shape[0], basic_img.shape[1]))

        opacity = 0.4
        superimposed_img = np.array([
                heatmap[:,:,0] * opacity + basic_img,
                heatmap[:,:,1] * opacity + basic_img,
                heatmap[:,:,2] * opacity + basic_img])

        superimposed_img = superimposed_img.transpose(1,2,0)
        superimposed_img /= np.max(superimposed_img)

        return superimposed_img

    def show(self, filename, image, label, cam):
        fig=plt.figure(figsize=(15, 20), frameon=False, constrained_layout=True)
        plt.axis('off')
        plt.title('\n' + filename.split('/')[-1][:-5])

        fig.add_subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.text(0.3, 0.3, label, size=15, color='black',
            ha="left", va="top",
            bbox=dict(boxstyle="round",
                ec=(1., 0.5, 0.5),
                fc=(1., 0.8, 0.8)))
        plt.axis('off')

        fig.add_subplot(1, 2, 2)
        plt.imshow(cam)
        plt.axis('off')

        fig.set_constrained_layout_pads(w_pad=0, h_pad=0,
            hspace=0, wspace=0)
        plt.show()

    def test(self, filename):
        image = imread(self.eval_dir + filename)
        image = image / 255

        if (image.ndim == 3):
            image = rgb2gray(image)

        preprocess_img = np.expand_dims(resize(image, (224, 224, 1)), axis=0)
        label = self.add_prob(preprocess_img)

        cam = self.class_activation_map(image, preprocess_img)
        self.show(filename, image, label, cam)
        print(filename, label)


    def test_all(self):
        tester.test('PNEUMONIA/person1946_bacteria_4874.jpeg')
        tester.test('NORMAL/NORMAL2-IM-1427-0001.jpeg')
        tester.test('PNEUMONIA/person1946_bacteria_4875.jpeg')
        tester.test('NORMAL/NORMAL2-IM-1430-0001.jpeg')
        tester.test('PNEUMONIA/person1947_bacteria_4876.jpeg')
        tester.test('NORMAL/NORMAL2-IM-1431-0001.jpeg')
        tester.test('PNEUMONIA/person1949_bacteria_4880.jpeg')
        tester.test('NORMAL/NORMAL2-IM-1436-0001.jpeg')
        tester.test('PNEUMONIA/person1950_bacteria_4881.jpeg')
        tester.test('NORMAL/NORMAL2-IM-1437-0001.jpeg')
        tester.test('PNEUMONIA/person1951_bacteria_4882.jpeg')
        tester.test('NORMAL/NORMAL2-IM-1438-0001.jpeg')
        tester.test('PNEUMONIA/person1952_bacteria_4883.jpeg')
        tester.test('NORMAL/NORMAL2-IM-1440-0001.jpeg')
        tester.test('PNEUMONIA/person1954_bacteria_4886.jpeg')
        tester.test('NORMAL/NORMAL2-IM-1442-0001.jpeg')


if __name__ == '__main__':
    tester = evaluateModel()
    tester.test_all()

