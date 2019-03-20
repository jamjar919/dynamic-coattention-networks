import tensorflow as tf
import numpy as np

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), axis= 2))
    length = tf.reduce_sum(used, axis=1)
    length = tf.cast(length, tf.int32)
    return length

def transpose(tensor):
    return tf.transpose(tensor,perm=[0,2,1])

def encoder(questions,contexts,embedding,hidden_unit_size=200,embedding_vector_size=300):
    '''
        Build the model for the document encoder
        questions: Tensor of questions
        contexts: Tensor of contexts
        embedding: Mappings from encoded questions to GLoVE vectors
    '''
    #dropout_rate = 0.3 # https://openreview.net/forum?id=rJeKjwvclx Authors claim to use 0.3 rate. 
    batch_size = questions.shape[0]
    #contexts_size = contexts.shape[1].value
    questions_size = questions.shape[1].value

    print("Batch size", batch_size)
    assert questions.shape[0] == batch_size, "question shape doesn't match batch size"
    assert contexts.shape[0] == batch_size, "context shape doesn't match batch size"

    # Vectorise the contexts and questions
    # Format [batch, length, depth]  
    context_embedding = tf.map_fn(lambda x:  tf.nn.embedding_lookup(embedding, x), contexts, dtype=tf.float32)
    question_embedding = tf.map_fn(lambda x:  tf.nn.embedding_lookup(embedding, x), questions, dtype=tf.float32) # shape 10x33x300
    #context_vector = tf.nn.dropout(context_vector, keep_prob = dropout_rate) # Add dropout
    #question_vector = tf.nn.dropout(question_vector, keep_prob = dropout_rate)

    # https://stackoverflow.com/questions/48238113/tensorflow-dynamic-rnn-state/48239320#48239320
    context_embedding_length = length(context_embedding)
    question_embedding_length = length(question_embedding)
    
    lstm_enc = tf.nn.rnn_cell.LSTMCell(hidden_unit_size)


    context_encoding, _ = tf.nn.dynamic_rnn(lstm_enc, context_embedding, sequence_length = context_embedding_length, dtype=tf.float32)
    print("context encoding shape: ",context_encoding.shape)
    # Append sentinel vector
    # https://stackoverflow.com/questions/52789457/how-to-perform-np-append-type-operation-on-tensors-in-tensorflow
    sentinel_vec_context = tf.Variable(tf.zeros([1, hidden_unit_size]), dtype = tf.float32)
    sentinel_vec_context_batch = tf.stack([sentinel_vec_context] * batch_size)
    context_encoding = tf.concat([context_encoding, sentinel_vec_context_batch], axis  = 1 )
    print("Extended context encoding shape: ", context_encoding.shape)

    
    question_encoding, _ = tf.nn.dynamic_rnn(lstm_enc, question_embedding, sequence_length = question_embedding_length, dtype=tf.float32) 
    print("Question encoding shape: ", question_encoding.shape)   
    # Append sentinel vector 
    sentinel_vec_question = tf.Variable(tf.zeros([1,hidden_unit_size]), dtype = tf.float32)
    sentinel_vec_q_batch = tf.stack([sentinel_vec_question] * batch_size) 
    question_encoding = tf.concat([question_encoding, sentinel_vec_q_batch], axis = 1)
    print("Extended question encoding shape: ",question_encoding.shape)

    # assert question_encoding.shape == (batch_size, hidden_unit_size, questions.shape[1] + 1), "question encoding shape doesn't match (batch size, hidden unit size, max question length + 1) " + str(question_encoding.shape)
    # assert context_encoding.shape == (batch_size, hidden_unit_size, contexts.shape[1] + 1), "context encoding shape doesn't match (batch size, hidden unit size, max context length + 1) " + str(context_encoding.shape)

    # Append "non linear projection layer" on top of the question encoding
    # Q = tanh(W^{Q} Q' + b^{Q})
    W_q = tf.Variable(tf.random_uniform([hidden_unit_size, hidden_unit_size]), [hidden_unit_size, hidden_unit_size], dtype=tf.float32)
    b_q = tf.Variable(tf.random_uniform([questions_size+1, hidden_unit_size]),  [questions_size+1, hidden_unit_size], dtype=tf.float32)
    W_q_batch = tf.stack([W_q] * batch_size)
    b_q_batch = tf.stack([b_q] * batch_size)
    Q = tf.tanh(tf.matmul(question_encoding, W_q_batch) + b_q_batch)
    print("Q shape :", Q.shape)
    # assert Q.shape == (batch_size, hidden_unit_size, questions.shape[1] + 1), "Q shape doesn't match (batch_size, hidden_unit_size, max question length + 1)"+ str(Q.shape)

    L = tf.matmul(context_encoding, transpose(Q))
    print("L.shape : ",L.shape)
    # assert L.shape == (batch_size, contexts.shape[1] + 1,  questions.shape[1] + 1), "L shape doesn't match (batch_size, max context length + 1,  max question length + 1)" + str(L.shape)

    # attention weights for questions A^{Q} = softmax(L)
    A_q = tf.nn.softmax(L) # rowwise on the rows of the matrices in the tensor. 
    print("A_q.shape ",A_q.shape)
    # assert A_q.shape == (batch_size, contexts.shape[1] + 1,  questions.shape[1] + 1), "A_q shape doesn't match (batch_size, max context length + 1,  max question length + 1)" + str(A_q.shape)

    # attention weights for documents A^{D} = softmax(L')
    A_d = tf.nn.softmax(transpose(L))
    print("A_d.shape ",A_d.shape)
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
    # C_transpose = transpose(C)
    u_states, _ = tf.nn.bidirectional_dynamic_rnn(cell_bw=cell_bw,cell_fw=cell_fw,dtype=tf.float32,inputs = C)
    
    U = tf.concat(u_states, axis = 2) # 10x633x400
    #print("U SHAPE LINE 107: ", U.shape)
    U = U[:,:-1,:] 
    # U = tf.slice(U, begin = [0,1,0], size = [batch_size, contexts.shape[1], 2*hidden_unit_size]) # Make U to 10x632x400
    print("U.shape ",U.shape)
    #print("U SHAPE AFTER SLICE:", U.shape)
    assert U.shape == (batch_size, contexts.shape[1], 2 * hidden_unit_size), "C shape doesn't match (batch_size, 2 * hidden_unit_size, max context length)" + str(U)
    
    return U, context_embedding_length


if __name__ == "__main__":
    print("Running encoder by itself for debug purposes.")
    question_batch_placeholder = tf.placeholder(dtype=tf.int32, shape=[10, 40],
                                                name='question_batch')
    context_batch_placeholder = tf.placeholder(dtype=tf.int32, shape=[10, 632],
                                               name='context_batch')
    embedding = tf.placeholder(shape=[1542, 300],
                               dtype=tf.float32, name='embedding')

    encoder(question_batch_placeholder, context_batch_placeholder, embedding)
    