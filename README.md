# PneNet: A Basic Pneumonia Classifier
#### Jose Arreluce and Max Demers

### Setup
Download the dataset from: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

Unzip the directory, and __rename the outermost directory from "chest_xray" to "data"__

### Data Exploration
Run visualize.py to see basic dataset information

### Training
Run model.py to train the model
* Epochs, batch size, and model filename can be adjusted at the bottom of model.py.
* Training will automatically generate loss and accuracy metrics to be saved as pngs.

### Evaluation
Run evaluation.py to see the quality of the model and class activation maps
* All images in the "val" directory will be tested one at a time
