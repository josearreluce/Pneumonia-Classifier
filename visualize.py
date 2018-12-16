import re
import matplotlib.pyplot as plt
import numpy as np
from os import listdir


def data_details(d_type):
    """
    Given image data subdirectory type, gets information of image types available and
    creates a visualization of that information and prints the raw counts as well

    :d_type: the data subdirectory (train, test, or val)
    """
    infected = listdir('./data/' + d_type + '/PNEUMONIA/')
    norm = listdir('./data/' + d_type + '/NORMAL/')

    bac = [f for f in infected if re.match(r'.*_bacteria_.*.jpeg', f)]
    vir = [f for f in infected if re.match(r'.*_virus_.*.jpeg', f)]

    print(d_type)
    print('     Total Pneumonia', len(infected),' (Bacterial:', len(bac), ')(Viral:', len(vir), ')')
    print('     Total Normal', len(norm))
    print('     Total Training Images', len(infected) + len(norm))

    fig, ax = plt.subplots()
    index = np.arange(4)
    r = [0, 2]
    names = ['Normal', 'Pneumonia']
    bars1 = [len(norm), len(bac)]
    bars2 = [0, len(vir)]
    barWidth = 1

    b1 = plt.bar(r, bars1, color=['#FFD700', '#ff5700'], edgecolor='black', width=barWidth)
    b2 = plt.bar(r, bars2, bottom=bars1, color='#5700ff', edgecolor='black', width=barWidth)

    ax.set_xlabel(d_type + ' Data Labels')
    ax.set_ylabel('X-Ray Image Count')
    ax.set_title(d_type + ' Data Details')
    plt.xticks(r, names, fontweight='bold')
    labels = ('Normal ({})'.format(len(norm)), 'Bacterial ({})'.format(len(bac)), 'Viral ({})'.format(len(vir)))

    plt.legend((b1[0], b1[1], b2[0]), labels)

    fig.tight_layout()
    plt.show()


def plot_training(tracker):
    """
    plots the training accuracy and loss over the span of the training session,
    as well as the testing accuracy and loss
    """
    # Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(tracker.history['loss'],'r',linewidth=3.0)
    plt.plot(tracker.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.savefig('training.png')

    # Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(tracker.history['acc'],'r',linewidth=3.0)
    plt.plot(tracker.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.savefig('testing.png')

def plot_training_alt(tracker, epochs):
    """
    plots the training accuracy and loss over the span of the training session,
    as well as the testing accuracy and loss, but does it all in one plot
    """
    # plot the training loss and accuracy
    plt.style.use('ggplot')
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), tracker.history['loss'], label='training loss')
    plt.plot(np.arange(0, N), tracker.history['acc'], label='training acc')
    plt.plot(np.arange(0, N), tracker.history['val_loss'], label='eval loss')
    plt.plot(np.arange(0, N), tracker.history['val_acc'], label='eval acc')
    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss and Accuracy')
    plt.legend(loc='center right')
    plt.savefig('training_alt.png')

if __name__ == '__main__':
    data_details('train')
    data_details('test')
    data_details('val')

