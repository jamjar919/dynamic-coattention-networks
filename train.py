# This file trains the neural network using the encoder and decoder.
import sys
import numpy as np
import tensorflow as tf
from functools import reduce

# custom imports
from encoder import encoder
from decoder import decoder
from dataset import Dataset

D = Dataset('data/dev.json', 'data/glove.6B.300d.txt')
padded_data, index2embedding, max_length_question, max_length_context = D.load_data(sys.argv[1:])
print("Loaded data")

# Train now
batch_size = 10
tf.reset_default_graph()
embedding = tf.Variable(index2embedding, dtype=tf.float32, trainable = False)
question_batch_placeholder = tf.placeholder(dtype=tf.int32, shape = [batch_size, max_length_question])
context_batch_placeholder = tf.placeholder(dtype=tf.int32, shape = [batch_size, max_length_context])

# Create encoder
U = encoder(question_batch_placeholder,context_batch_placeholder,embedding)

# Word index placeholders
answer_start = tf.placeholder(dtype=tf.int32,shape=[None])
answer_end = tf.placeholder(dtype=tf.int32,shape=[None])

# Create decoder 
s, e, s_logits, e_logits = decoder(U)

l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_start,logits=s_logits)
l2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_end,logits=e_logits)
loss = l1 + l2
train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print("SESSION INITIALIZED")
    for counter in range(0,101,batch_size):
        # running on an example batch to debug encoder
        batch = padded_data[counter:(counter+batch_size)]
        print(batch)
        print(batch.shape)
        question_batch = np.array(list(map(lambda qas: (qas["question"]), batch))).reshape(batch_size,max_length_question)
        context_batch = np.array(list(map(lambda qas: (qas["context"]), batch))).reshape(batch_size,max_length_context)
        answer_start_batch = np.array(list(map(lambda qas: (qas["answer_start"]), batch))).reshape(batch_size)
        answer_end_batch = np.array(list(map(lambda qas: (qas["answer_end"]), batch))).reshape(batch_size)
        print("BEFORE ENCODER RUN counter = ",counter)
        loss_val = sess.run(train_op,feed_dict = {
            question_batch_placeholder : question_batch,
            context_batch_placeholder : context_batch,
            answer_start : answer_start_batch,
            answer_end : answer_end_batch
        })
        print("loss: ",loss_val)
        counter += batch_size%len(padded_data)
        

