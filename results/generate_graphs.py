import pickle
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

path = os.path.dirname(os.path.abspath(__file__))

# Load saved data from the files
loss_averages_file = path+"/training_loss_means.pkl"
loss_full_csv = path+'/training_loss_per_batch.csv'
validation_f1_file = path+"./validation_f1_means.pkl"
validation_loss_file = path+"./validation_loss_means.pkl"

with open(loss_averages_file, "rb") as f:
    loss_averages = pickle.load(f)

with open(loss_full_csv, 'r') as f:
    losses = []
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        losses.append(list(map(lambda x : float(x), row)))

with open(validation_f1_file, "rb") as f:
    validation_f1_averages = pickle.load(f)

with open(validation_loss_file, "rb") as f:
    validation_loss_averages = pickle.load(f)

def generate_training_loss_graph():
    plt.clf()

    batches_per_epoch = len(losses[0])
    x_start = 0
    x_end = batches_per_epoch

    for epoch in range(0, len(losses)):
        plt.plot(list(range(x_start, x_end)), losses[epoch], 'b-', label="Loss in epoch "+str(epoch))
        x_start += batches_per_epoch
        x_end += batches_per_epoch

    current_x = 0
    current_x += batches_per_epoch / 2
    mean_x_coords = []
    for epoch in range(0, len(losses)):
        mean_x_coords.append(current_x)
        current_x += batches_per_epoch

    plt.plot(mean_x_coords, loss_averages, 'r-', label="Loss in epoch "+str(epoch))

    plt.title("Training Cross Entropy Loss over Batches Trained")
    plt.xlabel("Batches")
    plt.ylabel("Loss")

    plt.savefig(path + '/loss_graph.png')

    plt.clf()

def generate_training_validation_vs_loss():
    plt.clf()

    host = host_subplot(111, axes_class=AA.Axes)
    par = host.twinx()
    par.set_ylabel("F1 Score")
    par.set_ylim(0, 1)
    par.axis["right"].toggle(all=True)
    host.set_ylim(0, max(loss_averages) + 1)

    plt.plot(list(range(0, len(loss_averages))), loss_averages, 'b-', label="Training loss averages")
    plt.plot(list(range(0, len(loss_averages))), validation_loss_averages, 'r-', label="Validation loss averages")
    par.plot(list(range(0, len(loss_averages))), validation_f1_averages, 'y-', label="Validation f1 averages")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss versus Validation Loss")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00))

    plt.draw()
    plt.savefig(path + '/loss_validation_loss_graph.png')

    plt.clf()




generate_training_validation_vs_loss()
generate_training_loss_graph()