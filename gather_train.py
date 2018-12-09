from os import listdir
from scipy.misc import imread
from skimage.transform import resize
import json

pneumonia_training_directory = './data/train/PNEUMONIA/'
normal_training_directory = './data/train/NORMAL/'

training_images = []
training_labels = []
print("Gathering Training Images")
pcount = 0
for filename in listdir(pneumonia_training_directory):
    image = imread(pneumonia_training_directory + filename)
    image = resize(image, (224, 224))
    image = image.tolist()

    training_images.append(image)
    training_labels.append(1)

print("FINISHED GATHERING PNEUMONIA")
ncount = 0
for filename in listdir(normal_training_directory):
    image = imread(normal_training_directory + filename)
    image = resize(image, (224, 224))
    image = image.tolist()

    training_images.append(image)
    training_labels.append(0)
print("FINISHED GATHERING NORMAL")

data = {'images': training_images, 'labels': training_labels}
print(len(training_images))
with open('train_data.txt', 'w') as outfile:
    json.dump(data, outfile)
