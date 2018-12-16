import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from os import listdir
from skimage.transform import resize
from tensorflow.train import AdamOptimizer
from tensorflow.keras import backend
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import model_from_json
from scipy.misc import imread


class evaluateModel():
    def __init__(self, filename='model'):
        self.model = self.load_model(filename=filename)
        self.eval_dir = './data/val/'
        # The shape of the image when passed to the model
        self.img_shape = (224, 224, 1)


    def load_model(self, filename='model'):
        """
        Loads the model and trained weights created by train_model.py
        """
        with open(filename + '.json', 'r') as json_file:
            loaded_model_json = json_file.read()

        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(filename + '.h5')
        print('Loaded model')

        loaded_model.compile(optimizer=AdamOptimizer(),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        print(loaded_model.summary())
        return loaded_model


    def add_prob(self, image):
        """
        Given an image in proper input format, predict whether it is healthy or infected with
        pneumonia, and return a label indicating so.

        :return: a label string indicating pneumonia infection or not with the model's probability
        """
        prediction = self.model.predict(image)
        sick = prediction[0,0]
        norm = 1 - sick
        proba, label = (norm, 'Not Infected') if norm > sick else (sick, 'Pneumonia')
        label = '{}: {}%'.format(label, round(proba * 100, 2))
        return label


    def rgb2gray(self, rgb):
        """
        From HW2 Assignment
        Given an image in RGB format, convert image to one channel

        :return: an image in grayscale
        """
        gray = np.dot(rgb[...,:3],[0.29894, 0.58704, 0.11402])
        return gray


    def class_activation_map(self, basic_img, img):
        """
        Based on: https://jacobgil.github.io/deeplearning/class-activation-maps

        Creates a class activation heatmap given based on the loaded model, formats the heatmap
        to back to the original dimensions, and creates a superimposed image of the heatmap on
        the original image

        :unformatted_image: a numpy array of the image, unformatted (for superimposing)
        :formatted_image: A formatted numpy array of the image, ready to be passed to the model

        :return: A superimposed image depicting class activation based on the model
        """
        preds = self.model.predict(img)
        output = self.model.output[:, 0]

        # Get last convolutional layer from the model
        last_conv_layer = self.model.get_layer('conv2d_5')#'final_model_conv')
        # Create gradient of the conv layer vs. the final output
        grads = backend.gradients(output, last_conv_layer.output)[0]
        mean_grads = backend.mean(grads, axis=(0, 1, 2))

        # Function to grab the mean gradient and the raw conv layer output given model input
        get_cam_vals = backend.function([self.model.input], [mean_grads, last_conv_layer.output[0]])
        grads_val, final_conv_val = get_cam_vals([img])

        for i in range(last_conv_layer.output.shape[-1]):
            final_conv_val[:, :, i] *= grads_val[i]

        # Normalize output
        cam = np.mean(final_conv_val, axis=-1)
        cam /= np.max(cam)

        mapping = cm.get_cmap('rainbow_r')
        heatmap = mapping(np.uint8(255 * cam))
        #heatmap[np.where(cam > 0.5)] = 0 # thresholding

        # Convert heatmap back to original image dimensions
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
        """
        Given filename, raw image, and the calculated label and classification activation map,
        displays the image next to the CAM for visual analysis

        :filename: title of the display, used to verify proper labeling
        :image: raw image in a numpy array format
        :label: predicted label string to be added to the raw image
        :cam: The superimposed class activation mapping
        """
        fig=plt.figure(figsize=(15, 20), frameon=False, constrained_layout=True)
        plt.axis('off')
        plt.title('\n' + filename.split('/')[-1][:-5])

        fig.add_subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.text(0.3, 0.3, label, size=15, color='black',
            ha='left', va='top',
            bbox=dict(boxstyle='round',
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
        """
        Given a filename, generates a prediction label, a class activation map,
        and displays them. Also prints the label.

        :filename: file to open in the val dir
        """
        image = imread(self.eval_dir + filename)
        image = image / 255

        if (image.ndim == 3):
            image = rgb2gray(image)

        preprocess_img = np.expand_dims(resize(image, self.img_shape), axis=0)
        label = self.add_prob(preprocess_img)

        cam = self.class_activation_map(image, preprocess_img)
        self.show(filename, image, label, cam)
        print(filename, label)


    def test_all(self):
        """
        Runs test (above) for all images in val. Alternates infected and healthy examples
        """
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
    tester = evaluateModel(filename='model')
    tester.test_all()

