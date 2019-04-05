# This file runs the actual question answering program using our trained network
import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

from network.config import CONFIG
from dataset import Dataset
from preprocessing import answer_span_to_indices

# Suppress tensorflow verboseness
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def linear(x, x_max):
    return x/x_max

#https://www.desmos.com/calculator/osvpzhfmvp
def exponential(x):
    return np.exp(np.e * (x - 1))

def visualise_hwn(s, e, s_logits, e_logits, qas, dataset, filename="question.png"):
    path = os.path.dirname(os.path.abspath(__file__))
    plt.clf()

    context = []
    s_logits_filtered = []
    e_logits_filtered = []
    for i in range(0, len(s_logits)):
        if qas["context_mask"][i]:
            context.append(qas["context"][i])
            s_logits_filtered.append(s_logits[i])
            e_logits_filtered.append(e_logits[i])
    context = D.index_to_text(context).split()

    fig, axes = plt.subplots(nrows=1, ncols=len(s_logits_filtered))
    fig.set_size_inches(len(s_logits_filtered), 2)

    max_start = np.max(s_logits_filtered)
    max_end = np.max(e_logits_filtered)

    startcolors = np.zeros(shape=(len(s_logits), 3), dtype=np.float32)
    endcolors = np.zeros(shape=(len(e_logits), 3), dtype=np.float32)

    for i in range(0, len(s_logits_filtered)):
        startcolors[i][0] = exponential(linear(s_logits_filtered[i], max_start))
    
    for i in range(0, len(e_logits_filtered)):
        endcolors[i][2] = exponential(linear(e_logits_filtered[i], max_end))

    image_size = 10
    images = np.zeros((len(e_logits_filtered), image_size, image_size, 3))
    for i in range(0, len(e_logits_filtered)):
        for y in range(0, image_size):
            for x in range(0, int(image_size/2)):
                images[i][x][y] = startcolors[i]
            for x in range(int(image_size/2), image_size):
                images[i][x][y] = endcolors[i]

    for i in range(0, len(s_logits_filtered)):
        axes[i].imshow(images[i])
        axes[i].set_xlabel(context[i])
        axes[i].get_yaxis().set_visible(False)

    
    plt.subplots_adjust(wspace = 0)
    fig.suptitle(
        D.index_to_text(qas["question"]) + 
        "  ->  " + 
        D.index_to_text(answer_span_to_indices(s, e, qas["context"])) + 
        "  (real: " +
        D.index_to_text(answer_span_to_indices(qas["answer_start"], qas["answer_end"], qas["context"])) +
        ")", size=20
    )
    
    print("saving image...")
    plt.draw()
    plt.savefig(path + '/' +filename)
    plt.clf()
    print("done")

D = Dataset(CONFIG.EMBEDDING_FILE)
index2embedding = D.index2embedding
padded_data, (max_length_question, max_length_context) = D.load_questions(CONFIG.QUESTION_FILE)

random_question = np.random.choice(padded_data)

print("context:", D.index_to_text(random_question["context"]))
embedding_dimension = 300
init = tf.global_variables_initializer()

latest_checkpoint_path = tf.train.latest_checkpoint('./model/')
print("restoring from "+latest_checkpoint_path)
saver = tf.train.import_meta_graph(latest_checkpoint_path+'.meta')

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, latest_checkpoint_path)
    graph = tf.get_default_graph()
    s = graph.get_tensor_by_name("answer_start:0")
    e = graph.get_tensor_by_name("answer_end:0")
    alphas = graph.get_tensor_by_name("alphas:0")
    betas = graph.get_tensor_by_name("betas:0")
    question_batch_placeholder = graph.get_tensor_by_name("question_batch_ph:0")
    context_batch_placeholder = graph.get_tensor_by_name("context_batch_ph:0")
    embedding = graph.get_tensor_by_name("embedding_ph:0")
    dropout_keep_rate = graph.get_tensor_by_name("dropout_keep_ph:0")

    question = np.array([random_question["question"]] * CONFIG.BATCH_SIZE, dtype = np.int32)
    context = np.array([random_question["context"]] * CONFIG.BATCH_SIZE, dtype = np.int32)

    s_result, e_result, s_logits, e_logits = sess.run([s, e, alphas, betas], feed_dict = {
        question_batch_placeholder : question,
        context_batch_placeholder : context,
        embedding: index2embedding,
        dropout_keep_rate: 1
    })

    s_result = int(np.median(s_result))
    e_result = int(np.median(e_result))
    answer = answer_span_to_indices(s_result, e_result, random_question["context"])

    print()
    print(D.index_to_text(random_question["question"]), " -> ", D.index_to_text(answer))

    visualise_hwn(s_result, e_result, s_logits[-1][0], e_logits[-1][0], random_question, D)

    