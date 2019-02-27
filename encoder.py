import tensorflow as tf
from tensorflow.python.ops import math_ops
from keras import layers

# https://github.com/marshmelloX/dynamic-coattention-network/blob/master/selector.py
def encoder(questions,contexts,embedding,hidden_units_size=200):
    '''
        Build the model for the document encoder
        questions: Tensor of questions
        contexts: Tensor of contexts
        embedding: Mappings from encoded questions to GLoVE vectors
    '''
    batch_size = questions.get_shape()[0]

    print("Batch size", batch_size)
    print("Shape of questions", questions.get_shape())
    print("Shape of contexts", contexts.get_shape())

    with tf.variable_scope('embedding') as scope:
        # Vectorise the contexts and questions
        # Format [batch, length, depth]
        context_vector = tf.map_fn(lambda x:  tf.nn.embedding_lookup(embedding, x), contexts, dtype=tf.float32)
        question_vector = tf.map_fn(lambda x:  tf.nn.embedding_lookup(embedding, x), questions, dtype=tf.float32)

        context_embedding = tf.transpose(context_vector, perm=[1, 0, 2])
        question_embedding = tf.transpose(question_vector, perm=[1, 0, 2])

        lstm_enc = tf.nn.rnn_cell.LSTMCell(hidden_units_size)

    with tf.variable_scope('context_embedding') as scope:
        context_encoding, _ = tf.nn.dynamic_rnn(lstm_enc, context_embedding, dtype=tf.float32)
        # TODO append sentinel here 
    
    with tf.variable_scope('question_embedding') as scope:
        question_encoding, _ = tf.nn.dynamic_rnn(lstm_enc, question_embedding, dtype=tf.float32)
        # TODO append sentinel here

        # Append "non linear projection layer" on top of the question encoding
        # Q = tanh(W^{Q} Q' + b^{Q})
        # Essentially more weights and more biases, yay.
        print(question_encoding)
        question_weights = tf.Variable(tf.random_uniform([hidden_units_size, hidden_units_size]), [hidden_units_size, hidden_units_size], dtype=tf.float32)
        question_biases = tf.Variable(tf.random_uniform([hidden_units_size, hidden_units_size]),  [hidden_units_size, hidden_units_size], dtype=tf.float32)
        question_encoding = tf.map_fn(lambda x: math_ops.add(
            math_ops.matmul(question_weights, x),
            question_biases
        ), question_encoding, dtype=tf.float32)
        question_encoding = tf.tanh(question_encoding)
        print(question_encoding)
        # TODO fix this 

 
