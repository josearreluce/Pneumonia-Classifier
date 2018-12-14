import numpy as np
import matplotlib.pyplot as plt
import timeit
import os
import glob
import re



def training_data_details(pneumonia, healthy):
    sick = os.listdir(pneumonia)
    norm = os.listdir(healthy)

    bac = [f for f in sick if re.match(r'.*_bacteria_.*.jpeg', f)]
    vir = [f for f in sick if re.match(r'.*_virus_.*.jpeg', f)]

    print('Total Pneumonia', len(sick))
    print('     Bacterial:', len(bac))
    print('     Viral:', len(vir))
    print('Total Normal', len(norm))
    print('Total Training Images', len(sick) + len(norm))

    fig, ax = plt.subplots()
    index = np.arange(4)
    r = [0, 2]
    names = ['Normal', 'Pneumonia']
    bars1 = [len(norm), len(bac)]
    bars2 = [0, len(vir)]
    barWidth = 1

    b1 = plt.bar(r, bars1, color=['#FFD700', '#ff5700'], edgecolor='black', width=barWidth)
    b2 = plt.bar(r, bars2, bottom=bars1, color='#5700ff', edgecolor='black', width=barWidth)

    ax.set_xlabel('Training Data Labels')
    ax.set_ylabel('X-Ray Image Count')
    ax.set_title('Training Data Details')
    plt.xticks(r, names, fontweight='bold')
    labels = ('Normal ({})'.format(len(norm)), 'Bacterial ({})'.format(len(bac)), 'Viral ({})'.format(len(vir)))

    plt.legend((b1[0], b1[1], b2[0]), labels)

    fig.tight_layout()
    plt.show()


def plot_training(tracker):
    # Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(tracker.history['loss'],'r',linewidth=3.0)
    plt.plot(tracker.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)

    # Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(tracker.history['acc'],'r',linewidth=3.0)
    plt.plot(tracker.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.show()

def plot_training_alt(tracker, epochs):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), tracker.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), tracker.history["acc"], label="train_acc")
    plt.title("Training Loss and Accuracy on Santa/Not Santa")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")

if __name__ == '__main__':
    training_data_details('./data/train/PNEUMONIA/', './data/train/NORMAL/')
    training_data_details('./data/test/PNEUMONIA/', './data/test/NORMAL/')
    training_data_details('./data/val/PNEUMONIA/','./data/val/NORMAL/')


