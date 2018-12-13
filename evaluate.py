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
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

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
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")

        loaded_model.compile(optimizer=tf.train.AdamOptimizer(),
                           #loss='sparse_categorical_crossentropy',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        return loaded_model

    def plot_heatmap(self):
        pass

    def add_prob(self, image):
        prediction = self.model.predict(image)
        sick = prediction[0,0]
        norm = 1 - sick
        # build the label
        label = "Not Infected" if norm > sick else "Pneumonia"
        proba = norm if norm > sick else sick
        label = "{}: {}%".format(label, round(proba * 100, 2))
        return label

    # from hw2 assignment
    def rgb2gray(self, rgb):
        gray = np.dot(rgb[...,:3],[0.29894, 0.58704, 0.11402])
        return gray

    def test(self, filename):
        image = imread(self.eval_dir + filename)
        image = image / 255
        if (image.ndim == 3):
            image = rgb2gray(image)

        preprocess_img = np.expand_dims(resize(image, (224, 224, 1)), axis=0)

        label = self.add_prob(preprocess_img)
        cam = self.cam2(image, preprocess_img)

        #cam = self.cam(preprocess_img)

        plt.figure()
        plt.title(filename.split('/')[-1][:-5] + '\n')
        plt.text(0.3, 0.3, label, size=15, color='black',
            ha="left", va="top",
            bbox=dict(boxstyle="round",
                ec=(1., 0.5, 0.5),
                fc=(1., 0.8, 0.8),
            )
        )
        plt.imshow(image, cmap='gray')
        plt.show()

    def cam2(self, basic_img, img):
        import matplotlib.image as mpimg
        from tensorflow.keras import backend as K

        preds = self.model.predict(img)
        argmax = np.argmax(preds[0])
        print(self.model.summary())

        output = self.model.output[:, argmax]
        last_conv_layer = self.model.get_layer('conv2d_5')
        print(last_conv_layer)
        grads = K.gradients(output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([self.model.input], [pooled_grads, last_conv_layer.output[0]])

        pooled_grads_value, conv_layer_output_value = iterate([img])
        for i in range(32):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        import cv2

        #heatmap = resize(heatmap, (basic_img.shape[0], basic_img.shape[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        plt.imshow(heatmap)

        basic_img = resize(basic_img,(basic_img.shape[0], basic_img.shape[1]))

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        hif = .8
        superimposed_img = heatmap #* hif + basic_img
        output = './output.jpeg'
        cv2.imwrite(output, superimposed_img)
        img=mpimg.imread(output)
        plt.imshow(heatmap)
        plt.axis('off')
        return None


if __name__ == '__main__':
    evaluateModel().test('PNEUMONIA/person1949_bacteria_4880.jpeg')


