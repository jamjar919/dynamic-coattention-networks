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
    #lower_mat = tf.Print(lower_mat, [lower_mat.shape], "Lower mat shape")
    right_mat = val_two * tf.ones(shape = [seq_lens[0], N - seq_lens[1]], dtype = tf.float32)
    #right_mat = tf.Print(right_mat, [right_mat.shape], "Right mat shape")   
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
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_enc, output_keep_prob= dropout_keep_rate)

    context_encoding, _ = tf.nn.dynamic_rnn(lstm_cell, context_embedding, sequence_length = context_embedding_length, dtype=tf.float32)
    print("context encoding shape: ",context_encoding.shape)
    # Prepend sentinel vector
    # https://stackoverflow.com/questions/52789457/how-to-perform-np-append-type-operation-on-tensors-in-tensorflow
    sentinel_vec_context = tf.get_variable("sentinel_context", shape = [1, hidden_unit_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
    #sentinel_vec_context = tf.Print(sentinel_vec_context, [sentinel_vec_context[0:7]], "Sentinel context vector")
    sentinel_vec_context_batch = tf.stack([sentinel_vec_context] * batch_size)
    context_encoding = tf.concat([sentinel_vec_context_batch, context_encoding], axis  = 1 )
    print("Extended context encoding shape: ", context_encoding.shape)

    
    question_encoding, _ = tf.nn.dynamic_rnn(lstm_cell, question_embedding, sequence_length = question_embedding_length, dtype=tf.float32) 
    print("Question encoding shape: ", question_encoding.shape)   
    # Prepend sentinel vector 
    sentinel_vec_question = tf.get_variable("sentinel_question", shape = [1, hidden_unit_size], initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
    #sentinel_vec_question = tf.Print(sentinel_vec_question, [sentinel_vec_question[0:7]], "Sentinel question vector")
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
    Q_bin_mask = get_mask(question_embedding_length + 1, CONFIG.MAX_QUESTION_LENGTH + 1, hidden_unit_size, val_one = 1, val_two = 0)
    #Q = tf.Print(Q, [Q[0][1]], " Q SHAPE first row before mask")
    #Q = tf.Print(Q, [Q[0][-1]], " Q SHAPE last row before mask last rows")
    Q = Q * Q_bin_mask
    #Q = tf.Print(Q, [Q], " Q SHAPE first row after mask", summarize = 500)
    #Q = tf.Print(Q, [Q[0][-1]], " Q SHAPE last row", summarize = 500)
    print("Q shape :", Q.shape) # B,41,200

    # assert Q.shape == (batch_size, hidden_unit_size, questions.shape[1] + 1), "Q shape doesn't match (batch_size, hidden_unit_size, max question length + 1)"+ str(Q.shape)

    L = tf.matmul(context_encoding, transpose(Q))
    #L = tf.Print(L, [L[0, 0,:]], "Last row of L batch element", summarize = 200)
    #L = tf.Print(L, [L[0, context_embedding_length[0]+1:,:]], "First padded L row", summarize = 200)
    #L = tf.Print(L, [L[0, : , question_embedding_length[0]]], "last valid element of L question", summarize = 600)
    #L = tf.Print(L, [L[0, : , question_embedding_length[0]+1]], "First padded element of question", summarize = 600)
    L_mask = get_mask2D(tf.expand_dims(context_embedding_length + 1, -1), tf.expand_dims(question_embedding_length + 1, -1), CONFIG.MAX_CONTEXT_LENGTH + 1, CONFIG.MAX_QUESTION_LENGTH + 1, val_one = 0, val_two = -10**30)
    #L_mask = tf.Print(L_mask, [L_mask], "L mask", summarize = 1000)
    print("L.shape : ", L.shape)
    #print("L_mask: ", L_mask.shape)
    L = L + L_mask # Add ninf mask
    # assert L.shape == (batch_size, contexts.shape[1] + 1,  questions.shape[1] + 1), "L shape doesn't match (batch_size, max context length + 1,  max question length + 1)" + str(L.shape)
    #L = tf.Print(L, [L], "Carried out addition")
    # attention weights for questions A^{Q} = softmax(L)
    A_q = tf.nn.softmax(L) # rowwise on the rows of the matrices in the tensor. 
    print("A_q.shape ", A_q.shape)
    #A_q = tf.Print(A_q, [A_q[0, 0,:]], "A_q row values")
    # assert A_q.shape == (batch_size, contexts.shape[1] + 1,  questions.shape[1] + 1), "A_q shape doesn't match (batch_size, max context length + 1,  max question length + 1)" + str(A_q.shape)

    # attention weights for documents A^{D} = softmax(L')
    A_d = tf.nn.softmax(transpose(L))
    print("A_d.shape ", A_d.shape)
    # assert A_d.shape == (batch_size, questions.shape[1] + 1, contexts.shape[1] + 1), "A_d shape doesn't match (batch_size, max question length + 1, max context length + 1)" + str(A_d.shape)
    
    # Attention Context C^{Q}
    C_q = tf.matmul(transpose(A_q),context_encoding)
    print("C_q.shape :", C_q.shape)
    # assert C_q.shape == (batch_size, hidden_unit_size, questions.shape[1] + 1), "C_q shape doesn't match (batch_size, hidden_unit_size, max question length + 1)" + str(C_q)

    # C^{D} = [Q ; C^{Q}] A^{D}
    C_d = tf.matmul(transpose(A_d),tf.concat((Q,C_q), axis=2))
    print("C_d.shape: ",C_d.shape)
    # assert C_d.shape == (batch_size, 2 * hidden_unit_size, contexts.shape[1] + 1), "C_d shape doesn't match (batch_size, 2 * hidden_unit_size, max context length + 1)" + str(C_d)

    # Final context. Has no name in the paper, so we call it C
    C = tf.concat((context_encoding,C_d),axis=2)
    # assert C.shape == (batch_size, 3 * hidden_unit_size, contexts.shape[1] + 1), "C shape doesn't match (batch_size, 3 * hidden_unit_size, max context length + 1)" + str(C)
    
    print("C.shape : ",C.shape)

    # Bi-LSTM
    cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_unit_size)  
    cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_unit_size)
    (U1,U2), _ = tf.nn.bidirectional_dynamic_rnn(cell_bw=cell_bw,cell_fw=cell_fw, dtype=tf.float32,
        inputs = C, sequence_length = context_embedding_length + 1)
    print("U1 shape: ", U1.shape)
    U = tf.concat([U1,U2], axis = 2) # 10x633x400
    #U = tf.Print(U, [U[0,-1,:]], "U word 632")
    U = U[:,1:,:] 
    #U = tf.slice(U, begin = [0,1,0], size = [batch_size, contexts.shape[1], 2*hidden_unit_size]) # Make U to 10x632x400
    print("U.shape ", U.shape)
    #print("U SHAPE AFTER SLICE:", U.shape)
    #assert U.shape == (batch_size, contexts.shape[1], 2 * hidden_unit_size), "C shape doesn't match (batch_size, 2 * hidden_unit_size, max context length)" + str(U)
    
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

    encoder(question_batch_placeholder, context_batch_placeholder, 1.0, embedding)
    
