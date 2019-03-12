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

D = Dataset('data/dev.json', 'data/glove.6B.300d.txt')
padded_data, index2embedding, max_length_question, max_length_context = D.load_data(sys.argv[1:])
print("Loaded data")

# Train now
batch_size = 10
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
    for counter in range(0,101,batch_size):
        # running on an example batch to debug encoder
        batch = padded_data[counter:(counter+batch_size)]
        question_batch = np.array(list(map(lambda qas: (qas["question"]), batch))).reshape(batch_size,max_length_question)
        context_batch = np.array(list(map(lambda qas: (qas["context"]), batch))).reshape(batch_size,max_length_context)
        answer_start_batch = np.array(list(map(lambda qas: (qas["answer_start"]), batch))).reshape(batch_size)
        answer_end_batch = np.array(list(map(lambda qas: (qas["answer_end"]), batch))).reshape(batch_size)
        print("BEFORE ENCODER RUN counter = ",counter)
        sess.run(train_op,feed_dict = {
            question_batch_placeholder : question_batch,
            context_batch_placeholder : context_batch,
            answer_start : answer_start_batch,
            answer_end : answer_end_batch,
            embedding: index2embedding
        })
        loss_val = sess.run(loss,feed_dict = {
            question_batch_placeholder : question_batch,
            context_batch_placeholder : context_batch,
            answer_start : answer_start_batch,
            answer_end : answer_end_batch,
            embedding: index2embedding
        })
        print("loss: ",np.mean(loss_val))
        counter += batch_size%len(padded_data)
    '''
    tf.saved_model.simple_save(
        sess,
        '/model',
        inputs = {
            question_batch: question_batch_placeholder,
            context_batch: context_batch_placeholder
        },
        outputs = {
            answer_start: s,
            answer_end: e
        }
    )
    '''