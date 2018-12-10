import numpy as np
import matplotlib.pyplot as plt
import timeit
import os
import glob
import re

pneumonia_training_directory = './data/train/PNEUMONIA/'
normal_training_directory = './data/train/NORMAL/'


def training_data_details():
    sicc = os.listdir(pneumonia_training_directory)
    norm = os.listdir(normal_training_directory)

    bac = [f for f in sicc if re.match(r'.*_bacteria_.*.jpeg', f)]
    vir = [f for f in sicc if re.match(r'.*_virus_.*.jpeg', f)]

    print('Bacterial:', len(bac))
    print('Viral:', len(vir))
    print('Total Pneumonia', len(sicc))
    print('Total Normal', len(norm))
    print('Total Training Images', len(sicc) + len(norm))

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


if __name__ == '__main__':
    training_data_details()

