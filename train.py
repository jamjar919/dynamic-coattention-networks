# This file trains the neural network using the encoder and decoder.
import sys
import numpy as np
import tensorflow as tf
from functools import reduce

# custom imports
from encoder import encoder
from dataset import Dataset

D = Dataset('data/dev.json', 'data/glove.6B.300d.txt')
padded_data, index2embedding, max_length_question, max_length_context = D.load_data(sys.argv[1:])
print("Loaded data")


### Train now
batch_size = 10
tf.reset_default_graph()
embedding = tf.Variable(index2embedding, dtype=tf.float32, trainable = False)
question_batch_placeholder = tf.placeholder(dtype=tf.int32, shape = [batch_size, max_length_question])
context_batch_placeholder = tf.placeholder(dtype=tf.int32, shape = [batch_size, max_length_context])
U = encoder(question_batch_placeholder,context_batch_placeholder,embedding)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print("SESSION INITIALIZED")
    for counter in range(0,101,batch_size):
        # running on an example batch to debug encoder
        batch = padded_data[counter:(counter+batch_size)]
        question_batch = np.array(list(map(lambda qas: (qas["question"]), batch))).reshape(batch_size,max_length_question)
        context_batch = np.array(list(map(lambda qas: (qas["context"]), batch))).reshape(batch_size,max_length_context)
        print("BEFORE ENCODER RUN counter = ",counter)
        output = sess.run(U,feed_dict={question_batch_placeholder: question_batch,
            context_batch_placeholder: context_batch})
        print("AFTER ENCODER RUN counter = ",counter)
        counter += batch_size%len(padded_data)

