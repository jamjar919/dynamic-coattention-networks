import pickle
import os

import csv
import matplotlib
import sys
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import tensorflow as tf
path = os.path.dirname(os.path.abspath(__file__))

# Load saved data from the files
test_f1_file = path+"/testing_f1_means.pkl"
test_em_file = path+"/testing_em_means.pkl"
test_loss_file = path+"/testing_loss_means.pkl"

config = tf.ConfigProto()
if '--noGPU' in sys.argv[1:]:
    print("Not using the GPU...")
    config = tf.ConfigProto(device_count={'GPU': 0})

with open(test_f1_file, "rb") as f:
    testing_f1_averages = []
    for i in range(8):
        try:
            testing_f1_averages.append(pickle.load(f))
        except EOFError:
            break


with open(test_em_file, "rb") as f:
    test_em_averages = []
    for i in range(8):
        try:
            test_em_averages.append(pickle.load(f))
        except EOFError:
            break

with open(test_loss_file, "rb") as f:
    test_loss_averages = []
    for i in range(8):
        try:
            test_loss_averages.append(pickle.load(f))
        except EOFError:
            break

def generate_testing_losses():
    plt.clf()

    host = host_subplot(111, axes_class=AA.Axes)
    par = host.twinx()
    par.set_ylabel("F1 Score")
    par.set_ylim(0, 1)
    par.axis["right"].toggle(all=True)
    print(testing_f1_averages)
    print(test_em_averages)
    host.set_ylim(0, max(test_em_averages) + 1)

    plt.plot(list(range(0, len(testing_f1_averages))), test_em_averages, 'r-', label="Testing EM averages")
    par.plot(list(range(0, len(testing_f1_averages))), testing_f1_averages, 'y-', label="Testing F1 averages")

    plt.xlabel("Epochs")
    plt.ylabel("EM scores")
    plt.title("Testing EM and F1 vs epochs")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00))

    plt.draw()
    plt.savefig(path + '/testing_f1_em.png')

    plt.clf()


def test_losses_scores():
    plt.clf()

    host = host_subplot(111, axes_class=AA.Axes)
    par = host.twinx()
    par.set_ylabel("F1/EM Score")
    par.set_ylim(0, 1)
    par.axis["right"].toggle(all=True)
    host.set_ylim(0, max(test_loss_averages) + 1)
    host.set_xlim(0, len(test_loss_averages))

    plt.plot(list(range(1, len(test_loss_averages) + 1)), test_loss_averages, 'b-', label="Loss")
    par.plot(list(range(1, len(test_loss_averages) + 1)), testing_f1_averages, 'y-', label="F1 score")
    par.plot(list(range(1, len(test_loss_averages) + 1)), test_em_averages, 'g-', label="EM score")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Learning Curves on Dev Set")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00))

    plt.draw()
    plt.savefig(path + '/testing_graph.png')

    plt.clf()



#generate_testing_losses()
test_losses_scores()
