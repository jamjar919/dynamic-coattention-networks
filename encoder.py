import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.keras import layers
import numpy as np

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
    length = tf.reduce_sum(used, axis=1)
    length = tf.cast(length, tf.int32)
    return length

def transpose(tensor):
    return tf.transpose(tensor,perm=[0,2,1])

# https://github.com/marshmelloX/dynamic-coattention-network/blob/master/selector.py
def encoder(questions,contexts,embedding,hidden_units_size=300):
    '''
        Build the model for the document encoder
        questions: Tensor of questions
        contexts: Tensor of contexts
        embedding: Mappings from encoded questions to GLoVE vectors
    '''
    batch_size = questions.get_shape()[0]
    contexts_size = contexts.get_shape()[1].value
    questions_size = questions.get_shape()[1].value

    print("Batch size", batch_size)
    print("Shape of questions", questions.get_shape())
    print("Shape of contexts", contexts.get_shape())

    with tf.variable_scope('embedding') as scope:
        # Vectorise the contexts and questions
        # Format [batch, length, depth]
        context_vector = tf.map_fn(lambda x:  tf.nn.embedding_lookup(embedding, x), contexts, dtype=tf.float32)
        question_vector = tf.map_fn(lambda x:  tf.nn.embedding_lookup(embedding, x), questions, dtype=tf.float32)


        context_embedding = tf.transpose(context_vector, perm=[0, 2, 1])
        question_embedding = tf.transpose(question_vector, perm=[0, 2, 1])
        print("Context embedding shape : ",context_embedding.get_shape())
        print("Question embedding shape : ",question_embedding.get_shape())
        lstm_enc = tf.nn.rnn_cell.LSTMCell(hidden_units_size)

    with tf.variable_scope('context_embedding') as scope:
        # https://stackoverflow.com/questions/48238113/tensorflow-dynamic-rnn-state/48239320#48239320
        context_encoding, _ = tf.nn.dynamic_rnn(lstm_enc, transpose(context_embedding), sequence_length = length(context_embedding), dtype=tf.float32)
        context_encoding = transpose(context_encoding)
        # Append sentinel vector
        # https://stackoverflow.com/questions/52789457/how-to-perform-np-append-type-operation-on-tensors-in-tensorflow
        sentinel_vec = tf.constant(0, shape=[batch_size,hidden_units_size, 1], dtype = tf.float32)
        context_encoding = tf.concat((context_encoding, sentinel_vec), axis=-1)
        print("Context encoding shape : ",context_encoding.get_shape)

    with tf.variable_scope('question_embedding') as scope:
        question_encoding, _ = tf.nn.dynamic_rnn(lstm_enc, transpose(question_embedding), 
            sequence_length = length(question_embedding), dtype=tf.float32)
        question_encoding = transpose(question_encoding)
        sentinel_vec = tf.constant(0, shape=[batch_size,hidden_units_size, 1], dtype = tf.float32)
        question_encoding = tf.concat((question_encoding,sentinel_vec), axis = -1)
        print("Question encoding shape : ", question_encoding.get_shape())
    #     # Append "non linear projection layer" on top of the question encoding
    #     # Q = tanh(W^{Q} Q' + b^{Q})
    #     # Essentially more weights and more biases, yay.
    #     print(question_encoding)
    #     question_weights = tf.Variable(tf.random_uniform([hidden_units_size, hidden_units_size]), [hidden_units_size, hidden_units_size], dtype=tf.float32)
    #     question_biases = tf.Variable(tf.random_uniform([hidden_units_size, hidden_units_size]),  [hidden_units_size, hidden_units_size], dtype=tf.float32)
    #     question_encoding = tf.map_fn(lambda x: math_ops.add(
    #         math_ops.matmul(question_weights, x),
    #         question_biases
    #     ), question_encoding, dtype=tf.float32)
    #     question_encoding = tf.tanh(question_encoding)
    #     print(question_encoding)
    #     return question_encoding
    #     # TODO fix this 

    return context_encoding, length(context_embedding)

 
