import tensorflow as tf
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
def encoder(questions,contexts,embedding,embedding_size=300):
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

    with tf.variable_scope('context_embedding') as scope:
        lstm_enc = tf.nn.rnn_cell.LSTMCell(embedding_size)
        # https://stackoverflow.com/questions/48238113/tensorflow-dynamic-rnn-state/48239320#48239320
        context_encoding, _ = tf.nn.dynamic_rnn(lstm_enc, transpose(context_embedding), dtype=tf.float32)
        context_encoding = transpose(context_encoding)
        # Append sentinel vector
        # https://stackoverflow.com/questions/52789457/how-to-perform-np-append-type-operation-on-tensors-in-tensorflow
        sentinel_vec = tf.constant(0, shape=[batch_size,embedding_size, 1], dtype = tf.float32)
        context_encoding = tf.concat((context_encoding, sentinel_vec), axis=-1)
        print("Context encoding shape : ",context_encoding.get_shape())

    with tf.variable_scope('question_embedding') as scope:
        question_encoding, _ = tf.nn.dynamic_rnn(lstm_enc, transpose(question_embedding), dtype=tf.float32)
        question_encoding = transpose(question_encoding)
        # Append sentinel vector
        sentinel_vec = tf.constant(0, shape=[batch_size,embedding_size, 1], dtype = tf.float32)
        question_encoding = tf.concat((question_encoding,sentinel_vec), axis = -1)
        print("Question encoding shape : ", question_encoding.get_shape())
        # Append "non linear projection layer" on top of the question encoding
        # Q = tanh(W^{Q} Q' + b^{Q})
        W_q = tf.Variable(tf.random_uniform([embedding_size, embedding_size]), [embedding_size, embedding_size], dtype=tf.float32)
        b_q = tf.Variable(tf.random_uniform([embedding_size, questions_size+1]),  [embedding_size, questions_size+1], dtype=tf.float32)
        Q = tf.map_fn(lambda x: tf.math.add(
            tf.matmul(W_q, x),
            b_q
        ), question_encoding, dtype=tf.float32)
        Q = tf.tanh(question_encoding)
        print("Q shape: ",Q.get_shape())
        L = tf.matmul(transpose(context_encoding),Q)
        print("L shape: ",L.get_shape())
        # attention weights for questions A^{Q} = softmax(L)
        A_q = tf.nn.softmax(L)
        print("A_q shape: ",A_q.get_shape())
        # attention weights for documents A^{D} = softmax(L')
        A_d = tf.nn.softmax(transpose(L))
        print("A_d shape: ",A_d.get_shape())
        # Attention Context C^{Q}
        C_q = tf.matmul(context_encoding,A_q)
        print("C_q shape: ",C_q.get_shape())
        # C^{D} = [Q ; C^{Q}] A^{D}
        C_d = tf.matmul(tf.concat((Q,C_q), axis=1),A_d)
        print("C_d shape: ",C_d.get_shape())
        # Final context. Has no name in the paper, so we call it C
        C = tf.concat((context_encoding,C_d),axis=1)
        print("C shape: ",C.get_shape())
    
    with tf.variable_scope('coattention'):
        # Bi-LSTM
        cell_fw = tf.nn.rnn_cell.LSTMCell(embedding_size)  
        cell_bw = tf.nn.rnn_cell.LSTMCell(embedding_size)
        u_states, _ = tf.nn.bidirectional_dynamic_rnn(cell_bw=cell_bw,cell_fw=cell_fw,dtype=tf.float32,inputs= transpose(C))
        U = transpose(tf.concat(u_states,axis = 2))
        # Ignore u_{m+1}
        U = U[:,:,:-1]
        print("U shape: ",U.get_shape())
        return U


 
