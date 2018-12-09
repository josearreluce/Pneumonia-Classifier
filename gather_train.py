from os import listdir
from scipy.misc import imread
import json

pneumonia_training_directory = './data/train/PNEUMONIA/'
normal_training_directory = './data/train/NORMAL/'

training_images = []
training_labels = []
print("Gathering Training Images")
pcount = 0
for filename in listdir(pneumonia_training_directory):
    image = imread(pneumonia_training_directory + filename)
    #pneumonia_training_images.append((image, 1))
    training_images.append(image)
    training_labels.append(1)
    print(pcount)
    pcount += 1
print("FINISHED GATHERING PNEUMONIA")
ncount = 0
for filename in listdir(normal_training_directory):
    image = imread(normal_training_directory + filename)
    #normal_training_images.append((image, 0))
    training_images.append(image.tolist())
    training_labels.append(0)
    print(ncount)
    ncount += 1
print("FINISHED GATHERING NORMAL")
data = {'images': training_images, 'labels': training_labels}
#pickle.dumps((training_images, training_labels), open('train_data', "wb"), protocol=4)
with open('train_data.txt', 'w') as outfile:
    json.dump(data, outfile)

print("Finished Gathering Training Images")
