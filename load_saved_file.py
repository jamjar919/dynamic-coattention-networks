# This file runs the actual question answering program using our trained network
import sys
import numpy as np
import tensorflow as tf
import sklearn as sk
from functools import reduce

# custom imports
from dataset import Dataset

D = Dataset('data/dev.json', 'data/glove.6B.300d.txt')
padded_data, index2embedding, max_length_question, max_length_context = D.load_data(sys.argv[1:])
print("Loaded data")
print( index2embedding)
print(index2embedding.shape)


tf.reset_default_graph()
imported_graph = tf.train.import_meta_graph('trained-dcnn.meta')

init = tf.global_variables_initializer()
batch_size = 10

with tf.Session() as sess:

    imported_graph.restore(sess, 'trained-dcnn.meta')
    question_batch_placeholder = imported_graph.get_tensor_by_name("question_batch:0")
    context_batch_placeholder = imported_graph.get_tensor_by_name("context_batch:0")
    answer_start_batch_predict = imported_graph.get_tensor_by_name("answer_start_batch:0")
    answer_end_batch_predict = imported_graph.get_tensor_by_name("answer_end_batch:0")
    #question: question_batch, context: context_batch, answer_start: answer_start_batch, answer_end: answer_end_batch

    sess.run(init)
    print("SESSION INITIALIZED")
    for counter in range(0, 101, batch_size):
        # running on an example batch to debug encoder
        batch = padded_data[counter:(counter + batch_size)]
        question_batch = np.array(list(map(lambda qas: (qas["question"]), batch))).reshape(batch_size,
                                                                                           max_length_question)
        context_batch = np.array(list(map(lambda qas: (qas["context"]), batch))).reshape(batch_size, max_length_context)
        answer_start_batch_actual = np.array(list(map(lambda qas: (qas["answer_start"]), batch))).reshape(batch_size)
        answer_end_batch_actual = np.array(list(map(lambda qas: (qas["answer_end"]), batch))).reshape(batch_size)
        print("BEFORE ENCODER RUN counter = ", counter)
        loss_val = sess.run([answer_start_batch_predict, answer_end_batch_predict], feed_dict={
            question_batch_placeholder: question_batch,
            context_batch_placeholder: context_batch,
        })


        print(
        "Precision", sk.metrics.precision_score(
                np.concatenate(answer_start_batch_predict.eval(), answer_end_batch_predict.eval()),
                np.concatenate(answer_start_batch_actual.eval(), answer_end_batch_actual.eval())))
        print(
        "Recall", sk.metrics.recall_score(
                np.concatenate(answer_start_batch_predict.eval(), answer_end_batch_predict.eval()),
                np.concatenate(answer_start_batch_actual.eval(), answer_end_batch_actual.eval())))
        print(
        "f1_score", sk.metrics.f1_score(
                np.concatenate(answer_start_batch_predict.eval(), answer_end_batch_predict.eval()),
                np.concatenate(answer_start_batch_actual.eval(), answer_end_batch_actual.eval())))

