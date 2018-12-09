from os import listdir
from scipy.misc import imread
from skimage.transform import resize
import numpy as np

import json

test_images = []
test_labels = []

normal_test_directory = './data/test/NORMAL/'
pneumonia_test_directory = './data/test/PNEUMONIA/'

print("Gathering Test Images")
for filename in listdir(normal_test_directory):
    image = imread(normal_test_directory + filename)
    image = resize(image, (224, 224))
    image = image.tolist()

    test_images.append(image)
    test_labels.append(0)

for filename in listdir(pneumonia_test_directory):
    image = imread(pneumonia_test_directory + filename)
    image = resize(image, (224, 224))
    image = image.tolist()

    test_images.append(image)
    test_labels.append(1)

print("Finished Gathering Test Images")

#pickle.dump((test_images, test_labels), open('test_data', 'wb'))
data = {'images': test_images, 'labels': test_labels}
print(len(test_images))
with open('test_data.txt', 'w') as outfile:
    json.dump(data, outfile)
