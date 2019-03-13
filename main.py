# This file runs the actual question answering program using our trained network
import tensorflow as tf
import sys
import numpy as np

from dataset import Dataset

D = Dataset('data/dev.json', 'data/glove.6B.300d.txt')
padded_data, index2embedding, max_length_question, max_length_context = D.load_data(sys.argv[1:])

print(padded_data[0])
batch_size = 16
embedding_dimension = 300
init = tf.global_variables_initializer()

saver = tf.train.import_meta_graph('./model/saved.meta')
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, './model/saved')

    graph = tf.get_default_graph()
    s = graph.get_tensor_by_name("answer_start:0")
    e = graph.get_tensor_by_name("answer_end:0")
    question_batch_placeholder = graph.get_tensor_by_name("question_batch:0")
    context_batch_placeholder = graph.get_tensor_by_name("context_batch:0")
    embedding = graph.get_tensor_by_name("embedding:0")

    question = np.array([padded_data[0]["question"]] * batch_size, dtype = np.int32)
    context = np.array([padded_data[0]["context"]] * batch_size, dtype = np.int32)

    print(question.shape)
    print(context.shape)

    s_result, e_result = sess.run([s, e], feed_dict = {
        question_batch_placeholder : question,
        context_batch_placeholder : context,
        embedding: index2embedding
    })
    
    print(s_result, e_result)
