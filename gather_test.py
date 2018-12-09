from os import listdir
from scipy.misc import imread
import numpy as np
import pickle

test_images = []
test_labels = []

normal_test_directory = './data/test/NORMAL/'
pneumonia_test_directory = './data/test/PNEUMONIA/'

print("Gathering Test Images")
for filename in listdir(normal_test_directory):
    image = imread(normal_test_directory + filename)
    test_images.append(np.array(image))
    test_labels.append(0)

for filename in listdir(pneumonia_test_directory):
    image = imread(pneumonia_test_directory + filename)
    test_images.append(np.array(image))
    test_labels.append(1)

print("Finished Gathering Test Images")

pickle.dump((test_images, test_labels), open('test_data', 'wb'))