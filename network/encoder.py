import tensorflow as tf
import numpy as np
from config import CONFIG

def masked_matrix(seq_len,m,n,val_one = 1, val_two = 0) :
    val_one_matrix = val_one * tf.ones(shape = [seq_len, n], dtype = tf.float32)
    val_two_matrix = val_two * tf.ones(shape = [m - seq_len, n], dtype = tf.float32)
    
    return tf.concat(values = [val_one_matrix, val_two_matrix], axis = 0)

def get_mask(seq_lens, m,n, val_one = 0, val_two = 0):
    fn = lambda seq_len : masked_matrix(seq_len, m, n, val_one, val_two) 
    return tf.map_fn(fn,seq_lens ,dtype = tf.float32)

# M is total rows of mask. N total columns. m is seq length D, n is seq length Q.
def masked_matrix2d(seq_lens, M, N, val_one, val_two):
    one_mat = val_one * tf.ones(shape = [seq_lens[0], seq_lens[1]], dtype = tf.float32)
    lower_mat = val_two * tf.ones(shape = [M - seq_lens[0], N], dtype = tf.float32)
    right_mat = val_two * tf.ones(shape = [seq_lens[0], N - seq_lens[1]], dtype = tf.float32)
    upper_mat = tf.concat([one_mat, right_mat], axis = 1)
    full_mat = tf.concat([upper_mat, lower_mat], axis = 0)
    
    return full_mat

def get_mask2D(seq_D, seq_Q, M, N, val_one = 1, val_two = 0):
    seq_lens = tf.concat([seq_D, seq_Q], axis = 1)
    fn = lambda seq_len : masked_matrix2d(seq_len, M, N, val_one, val_two)
    return tf.map_fn(fn, seq_lens, dtype = tf.float32)

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), axis= 2))
    length = tf.reduce_sum(used, axis=1)
    length = tf.cast(length, tf.int32)
    return length

def transpose(tensor):
    return tf.transpose(tensor,perm=[0,2,1])

def encoder(questions,contexts,embedding, dropout_keep_rate):
    '''
        Build the model for the document encoder
        questions: Tensor of questions
        contexts: Tensor of contexts
        embedding: Mappings from encoded questions to GLoVE vectors
    '''
    #dropout_rate = 0.3 # https://openreview.net/forum?id=rJeKjwvclx Authors claim to use 0.3 rate. 
    batch_size = questions.shape[0].value
    #contexts_size = contexts.shape[1].value
    questions_size = questions.shape[1].value
    hidden_unit_size = CONFIG.HIDDEN_UNIT_SIZE
    embedding_dimension = CONFIG.EMBEDDING_DIMENSION

    print("Batch size", batch_size)

    # Vectorise the contexts and questions
    # Format [batch, length, depth]  
    context_embedding = tf.nn.embedding_lookup(embedding, contexts)
    question_embedding = tf.nn.embedding_lookup(embedding, questions)
    # https://stackoverflow.com/questions/48238113/tensorflow-dynamic-rnn-state/48239320#48239320
    context_embedding_length = length(context_embedding)
    question_embedding_length = length(question_embedding)
    print("Context embedding length shape: ", context_embedding_length.shape)
    
    lstm_enc = tf.nn.rnn_cell.LSTMCell(hidden_unit_size)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_enc, output_keep_prob = dropout_keep_rate)

    context_encoding, _ = tf.nn.dynamic_rnn(lstm_cell, context_embedding, sequence_length = context_embedding_length, dtype=tf.float32)
    print("context encoding shape: ",context_encoding.shape)
    # Prepend sentinel vector
    sentinel_vec_context = tf.get_variable("sentinel_context", shape = [1, hidden_unit_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
    sentinel_vec_context_batch = tf.stack([sentinel_vec_context] * batch_size)
    context_encoding = tf.concat([sentinel_vec_context_batch, context_encoding], axis  = 1 )
    context_encoding = tf.identity(context_encoding, name='context_encoding')
    print("Extended context encoding shape: ", context_encoding.shape)

    
    question_encoding, _ = tf.nn.dynamic_rnn(lstm_cell, question_embedding, sequence_length = question_embedding_length, dtype=tf.float32) 
    print("Question encoding shape: ", question_encoding.shape)   
    # Prepend sentinel vector 
    sentinel_vec_question = tf.get_variable("sentinel_question", shape = [1, hidden_unit_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
    sentinel_vec_q_batch = tf.stack([sentinel_vec_question] * batch_size) 
    question_encoding = tf.concat([sentinel_vec_q_batch, question_encoding], axis = 1)
    print("Extended question encoding shape: ",question_encoding.shape)

    # Append "non linear projection layer" on top of the question encoding
    # Q = tanh(W^{Q} Q' + b^{Q})
    W_q = tf.get_variable(name="W_q",shape=[hidden_unit_size,hidden_unit_size],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b_q = tf.get_variable(name= "b_q",shape= [questions_size+1, hidden_unit_size], initializer = tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    W_q_batch = tf.stack([W_q] * batch_size)
    b_q_batch = tf.stack([b_q] * batch_size)
    Q = tf.tanh(tf.add(tf.matmul(question_encoding, W_q_batch), b_q_batch))
    Q = tf.identity(Q, name='Q')
    Q_bin_mask = get_mask(question_embedding_length + 1, CONFIG.MAX_QUESTION_LENGTH + 1, hidden_unit_size, val_one = 1, val_two = 0)
    Q = Q * Q_bin_mask
    print("Q shape :", Q.shape) # B,41,200

    L = tf.matmul(context_encoding, transpose(Q))
    L_mask = get_mask2D(tf.expand_dims(context_embedding_length + 1, -1), tf.expand_dims(question_embedding_length + 1, -1), CONFIG.MAX_CONTEXT_LENGTH + 1, CONFIG.MAX_QUESTION_LENGTH + 1, val_one = 0, val_two = -10**30)
    print("L.shape : ", L.shape)
    L = L + L_mask # Add ninf mask
    A_q = tf.nn.softmax(L) # rowwise on the rows of the matrices in the tensor.
    A_q_mask = get_mask2D(tf.expand_dims(context_embedding_length + 1, -1), tf.expand_dims(question_embedding_length + 1, -1), CONFIG.MAX_CONTEXT_LENGTH + 1, CONFIG.MAX_QUESTION_LENGTH + 1, val_one = 1, val_two = 0)
    A_q = A_q * A_q_mask
    A_q = tf.identity(A_q, name='A_q')

    print("A_q.shape ", A_q.shape)
    
    A_d = tf.nn.softmax(transpose(L))
    A_d_mask = transpose(A_q_mask)
    A_d = A_d * A_d_mask 
    A_d = tf.identity(A_d, name='A_d')
    print("A_d.shape ", A_d.shape)
    
    # Attention Context C^{Q}
    C_q = tf.matmul(transpose(A_q),context_encoding)
    C_q = tf.identity(C_q, name='C_q')
    print("C_q.shape :", C_q.shape)

    C_d = tf.matmul(transpose(A_d),tf.concat((Q,C_q), axis=2))
    C_d = tf.identity(C_d, name='C_d')
    print("C_d.shape: ",C_d.shape)

    # Final context before BiLSTM. Has no name in the paper, so we call it C
    C = tf.concat((context_encoding,C_d),axis=2)
    C = tf.identity(C, name='C_d')
    
    print("C.shape : ",C.shape)

    # Bi-LSTM
    cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_unit_size) 
    cell_fw_dropout = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob = dropout_keep_rate)
    cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_unit_size)
    cell_bw_dropout = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob = dropout_keep_rate)
    (U1,U2), _ = tf.nn.bidirectional_dynamic_rnn(cell_bw=cell_bw_dropout, cell_fw=cell_fw_dropout, dtype=tf.float32,
        inputs = C, sequence_length = context_embedding_length + 1)
    print("U1 shape: ", U1.shape)
    U = tf.concat([U1,U2], axis = 2) # 10x633x400
    print("U.shape ", U.shape)
    return U, context_embedding_length



if __name__ == "__main__":
    print("Running encoder by itself for debug purposes.")
    question_batch_placeholder = tf.placeholder(dtype=tf.int32, shape=[10, 40],
                                                name='question_batch')
    context_batch_placeholder = tf.placeholder(dtype=tf.int32, shape=[10, 632],
                                               name='context_batch')
    embedding = tf.placeholder(shape=[1542, 300],
                               dtype=tf.float32, name='embedding')
    batch_size = 10

    encoder(question_batch_placeholder, context_batch_placeholder, embedding, 1.0)
    
