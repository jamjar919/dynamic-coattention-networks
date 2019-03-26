# This file runs the actual question answering program using our trained network
import tensorflow as tf
import sys
import numpy as np

from config import CONFIG
from dataset import Dataset
from preprocessing import answer_span_to_indices

D = Dataset(CONFIG.EMBEDDING_FILE)
index2embedding = D.index2embedding
padded_data, (max_length_question, max_length_context) = D.load_questions(CONFIG.QUESTION_FILE)

random_question = np.random.choice(padded_data)

print("context:", D.index_to_text(random_question["context"]))
batch_size = 16 # needs to match
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
    question_batch_placeholder = graph.get_tensor_by_name("question_batch_ph:0")
    context_batch_placeholder = graph.get_tensor_by_name("context_batch_ph:0")
    embedding = graph.get_tensor_by_name("embedding_ph:0")
    dropout_keep_rate = graph.get_tensor_by_name("dropout_keep_ph:0")

    question = np.array([random_question["question"]] * batch_size, dtype = np.int32)
    context = np.array([random_question["context"]] * batch_size, dtype = np.int32)

    s_result, e_result = sess.run([s, e], feed_dict = {
        question_batch_placeholder : question,
        context_batch_placeholder : context,
        embedding: index2embedding,
        dropout_keep_rate: 1
    })

    s_result = int(np.median(s_result))
    e_result = int(np.median(s_result))
    answer = answer_span_to_indices(s_result, e_result, random_question["context"])

    print()
    print(D.index_to_text(random_question["question"]), " -> ", D.index_to_text(answer))