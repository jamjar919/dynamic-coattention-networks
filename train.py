# This file trains the neural network using the encoder and decoder.
import sys
import numpy as np
import tensorflow as tf
from functools import reduce
import os

# custom imports
from encoder import encoder
from decoder import decoder
from dataset import Dataset

ITERATIONS = 40
tensorboard_filepath = '.'

D = Dataset('data/dev.json', 'data/glove.6B.300d.txt')
padded_data, index2embedding, max_length_question, max_length_context = D.load_data(sys.argv[1:])
print("Loaded data")

# Train now
batch_size = 16
embedding_dimension = 300
tf.reset_default_graph()

embedding = tf.placeholder(shape = [len(index2embedding), embedding_dimension], dtype=tf.float32, name='embedding')
question_batch_placeholder = tf.placeholder(dtype=tf.int32, shape = [batch_size, max_length_question], name='question_batch')
context_batch_placeholder = tf.placeholder(dtype=tf.int32, shape = [batch_size, max_length_context], name='context_batch')

# Create encoder
U = encoder(question_batch_placeholder,context_batch_placeholder,embedding)

# Word index placeholders
answer_start = tf.placeholder(dtype=tf.int32,shape=[None], name='answer_start_true')
answer_end = tf.placeholder(dtype=tf.int32,shape=[None], name='answer_end_true')

# Create decoder 
s, e, s_logits, e_logits = decoder(U)

s = tf.identity(s, name='answer_start')
e = tf.identity(e, name='answer_end')

l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_start,logits = s_logits)
l2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_end,logits = e_logits)

loss = l1 + l2
train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)


saver = tf.train.Saver() 

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print("SESSION INITIALIZED")
    dataset_size = len(padded_data)
    for epochs in range(ITERATIONS):
        # running on an example batch to debug encoder
        batch_rand_indices = np.random.choice(len(padded_data), batch_size)
        print("random batch indices :", batch_rand_indices)
        batch = np.array(padded_data)[batch_rand_indices]
        question_batch = np.array(list(map(lambda qas: (qas["question"]), batch))).reshape(batch_size,max_length_question)
        context_batch = np.array(list(map(lambda qas: (qas["context"]), batch))).reshape(batch_size,max_length_context)
        answer_start_batch = np.array(list(map(lambda qas: (qas["answer_start"]), batch))).reshape(batch_size)
        answer_end_batch = np.array(list(map(lambda qas: (qas["answer_end"]), batch))).reshape(batch_size)
        print("Epoch # : ", epochs)
        _ , loss_val = sess.run([train_op,loss],feed_dict = {
            question_batch_placeholder : question_batch,
            context_batch_placeholder : context_batch,
            answer_start : answer_start_batch,
            answer_end : answer_end_batch,
            embedding: index2embedding
        })
        tf.summary.histogram('loss', loss_val)
        print("loss: ",np.mean(loss_val))

    saver.save(sess, './model/saved') 

