# This file trains the neural network using the encoder and decoder.
import sys
import numpy as np
import tensorflow as tf
from functools import reduce

# custom imports
from encoder import encoder
from decoder import decoder
from dataset import Dataset

''' # NOT NEEDED IF WE ARE USING SPARSE CROSS ENTROPY LOSS. 
# Take in 1d batch labels and return 1 hot vector.
def toOneHot(batch, dim = 632):
    matrix = np.zeros((batch.shape[0], dim), dtype=np.int)
    for i in range (0, batch.shape[0]):
        matrix[i][batch[i] - 1] = 1
    return matrix
'''    

D = Dataset('data/dev.json', 'data/glove.6B.300d.txt')
padded_data, index2embedding, max_length_question, max_length_context = D.load_data(sys.argv[1:])
print("Loaded data")

# Train now
batch_size = 5
tf.reset_default_graph()
#embedding = tf.Variable(index2embedding, dtype=tf.float32, trainable = False)
embedding = tf.placeholder(dtype = tf.float32, shape = [400001, 300])
#embedding = tf.placeholer()
question_batch_placeholder = tf.placeholder(dtype=tf.int32, shape = [batch_size, max_length_question])
context_batch_placeholder = tf.placeholder(dtype=tf.int32, shape = [batch_size, max_length_context])

# Create encoder
U = encoder(question_batch_placeholder,context_batch_placeholder,embedding)

# Word index placeholders
#answer_start_lab = tf.placeholder(dtype=tf.int32,shape=[632, batch_size])
#answer_end_lab = tf.placeholder(dtype=tf.int32,shape=[632, batch_size])

answer_start = tf.placeholder(dtype=tf.int32,shape=[None])
answer_end = tf.placeholder(dtype=tf.int32,shape=[None])

# Create decoder 
s, e, s_logits, e_logits = decoder(U, answer_start, answer_end)

# Transpose for logits cause 1st dim of labels and logits MUST be equal. new logits shape: [batch_size, dim]
l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_start, logits = tf.transpose(s_logits))
l2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_end, logits = tf.transpose(e_logits))
loss = l1 + l2
train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print("SESSION INITIALIZED")
   # for counter in range(0,101,batch_size):
    # running on an example batch to debug encoder
    batch = padded_data[0:batch_size]
    
    question_batch = np.array(list(map(lambda qas: (qas["question"]), batch))).reshape(batch_size,max_length_question)
    print ("question_batch.shape: ", question_batch.shape)
    context_batch = np.array(list(map(lambda qas: (qas["context"]), batch))).reshape(batch_size,max_length_context)
    print ("context_batch.shape: ", context_batch.shape)
    answer_start_batch = np.array(list(map(lambda qas: (qas["answer_start"]), batch))).reshape(batch_size)
    #answer_start_batch_lab = toOneHot(answer_start_batch) NOT NEEDED SINCE USING SPARSE CE LOSS FUNCTION
    print ("answer_start_batch.shape: ", answer_start_batch.shape)
    answer_end_batch = np.array(list(map(lambda qas: (qas["answer_end"]), batch))).reshape(batch_size)
    #answer_end_batch_lab = toOneHot(answer_end_batch)
    print ("answer_end_batch.shape: ", answer_end_batch.shape)
    
    print("\n Running test session with one batch. ")
    
    loss = sess.run(train_op,feed_dict = {
        question_batch_placeholder : question_batch,
        context_batch_placeholder : context_batch,
        answer_start : answer_start_batch,
        answer_end : answer_end_batch,
        embedding : index2embedding
    })
    s = tf.print(s, [s], "Value of s: ")
    print ("Loss: ", loss)
    
    #counter += batch_size%len(padded_data)

