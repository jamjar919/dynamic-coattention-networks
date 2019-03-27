import matplotlib.pyplot as plt
import os 
import json
import numpy as np
path = os.path.dirname(os.path.abspath(__file__))

DEV_FILE_NAME = path + '/dev.json'
TRAIN_FILE_NAME = path + '/train.json'

with open(DEV_FILE_NAME, "r") as f:
    dev_data = json.loads(f.read())
    assert dev_data["version"] == "1.1"
    dev_data = np.array(dev_data["data"])

with open(TRAIN_FILE_NAME, "r") as f:
    train_data = json.loads(f.read())
    assert train_data["version"] == "1.1"
    train_data = np.array(train_data["data"])

def visualise_question_data(categories, stub=''):
    question_lengths = []
    context_lengths = []

    for category in categories:
        for paragraph in category["paragraphs"]:
            context_lengths.append(len(paragraph["context"].split(" ")))
            for qas in paragraph["qas"]:
                if len(qas["question"].split(" ")) > 100:
                    print(qas)
                question_lengths.append(len(qas["question"].split(" ")))
    
    plt.clf()

    fig = plt.figure()

    a=fig.add_subplot(1,2,1)
    plt.title(stub.capitalize() + " Question Distribution")
    plt.xlabel("Question Length")
    plt.ylabel("Frequency")
    plt.hist(question_lengths, bins=np.arange(min(question_lengths), max(question_lengths)+1))

    a=fig.add_subplot(1,2,2)
    plt.title(stub.capitalize() + " Context Distribution")
    plt.xlabel("Context Length")
    plt.ylabel("Frequency")
    plt.hist(context_lengths, bins=np.arange(min(context_lengths), max(context_lengths)+1))
    plt.subplots_adjust(wspace=0.3)

    plt.draw()
    plt.savefig(path + '/' + stub + '_question_histogram.png')

    plt.clf()

visualise_question_data(dev_data, 'dev')
visualise_question_data(train_data, 'train')
print("done")