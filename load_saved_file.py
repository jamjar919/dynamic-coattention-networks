# This file runs the actual question answering program using our trained network
import sys
import numpy as np
import tensorflow as tf
import sklearn as sk
from functools import reduce
from sklearn.metrics import precision_score, recall_score, f1_score

# custom imports
from dataset import Dataset

print("Starting testing...")
D = Dataset('data/dev.json', 'data/glove.6B.300d.txt')
padded_data, index2embedding, max_length_question, max_length_context = D.load_data(sys.argv[1:])
print("Loaded data")
print( index2embedding)
print(index2embedding.shape)


tf.reset_default_graph()
imported_graph = tf.train.import_meta_graph('./model/saved-6.meta')

init = tf.global_variables_initializer()
batch_size = 64

with tf.Session() as sess:

    imported_graph.restore(sess, './model/saved-6')
    graph = tf.get_default_graph()

    question_batch_placeholder = graph.get_tensor_by_name("question_batch:0")
    context_batch_placeholder = graph.get_tensor_by_name("context_batch:0")
    answer_start_batch_predict = graph.get_tensor_by_name("answer_start:0")
    answer_end_batch_predict = graph.get_tensor_by_name("answer_end:0")
    embedding = graph.get_tensor_by_name("embedding:0")

    sess.run(init)
    print("SESSION INITIALIZED")
    for counter in range(0, batch_size * 10, batch_size):
        # running on an example batch to debug encoder
        batch = padded_data[counter:(counter + batch_size)]
        question_batch = np.array(list(map(lambda qas: (qas["question"]), batch))).reshape(batch_size,
                                                                                           max_length_question)
        context_batch = np.array(list(map(lambda qas: (qas["context"]), batch))).reshape(batch_size, max_length_context)
        answer_start_batch_actual = np.array(list(map(lambda qas: (qas["answer_start"]), batch))).reshape(batch_size)
        answer_end_batch_actual = np.array(list(map(lambda qas: (qas["answer_end"]), batch))).reshape(batch_size)
        print("BEFORE ENCODER RUN counter = ", counter)
        s, e = sess.run([answer_start_batch_predict, answer_end_batch_predict], feed_dict={
            question_batch_placeholder: question_batch,
            context_batch_placeholder: context_batch,
            embedding: index2embedding
        })

        print(answer_start_batch_actual, answer_end_batch_actual)

        predictions = np.concatenate([s, e])
        actual = np.concatenate([answer_start_batch_actual, answer_end_batch_actual])

        print("predictions",predictions)
        print("actual", actual)

        print(
        "Precision", sk.metrics.precision_score(
                predictions,
                actual, average='micro')
        )
        print(
        "Recall", sk.metrics.recall_score(
                predictions,
                actual, average='micro')
        )
        print(
        "f1_score", sk.metrics.f1_score(
                predictions,
                actual, average='micro')
        )

