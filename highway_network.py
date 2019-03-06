import tensorflow as tf


HIDDEN_STATE_SIZE = 200 # named L in the paper
POOL_SIZE = 16

def transpose(tensor):
    return tf.transpose(tensor,perm=[0,2,1])

def highway_network_batch(U, lstm_hidden_state,
                    coattention_encoding_of_prev_start_word,
                    coattention_encoding_of_prev_end_word,
                    wd, w1, w2, w3,
                    b1, b2, b3):

    U_transpose = transpose(U)
    fn = lambda doc_encoding : highway_network_matrix(doc_encoding, lstm_hidden_state,
                    coattention_encoding_of_prev_start_word,
                    coattention_encoding_of_prev_end_word,
                    wd, w1, w2, w3,
                    b1, b2, b3)
    # returns 10 * 1
    return tf.map_fn(fn, U_transpose)

# U_transpose is of size 632 * 400
def highway_network_matrix(U_transpose, lstm_hidden_state,
                    coattention_encoding_of_prev_start_word,
                    coattention_encoding_of_prev_end_word,
                    wd, w1, w2, w3,
                    b1, b2, b3):

    #U_transpose = tf.transpose(U, perm=[1, 0])
    fn = lambda col : highway_network_single(col, lstm_hidden_state,
                    coattention_encoding_of_prev_start_word,
                    coattention_encoding_of_prev_end_word,
                    wd, w1, w2, w3,
                    b1, b2, b3)
    # returns 1 number
    return tf.reduce_max(tf.map_fn(fn, U_transpose))

def highway_network_single(coattention_encoding_of_word_in_doc,
                    lstm_hidden_state,
                    coattention_encoding_of_prev_start_word,
                    coattention_encoding_of_prev_end_word,
                    wd, w1, w2, w3,
                    b1, b2, b3):
    # weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    #wd = tf.get_variable("wd", shape=[HIDDEN_STATE_SIZE, 5 * HIDDEN_STATE_SIZE],
    #                     initializer=weight_initer)

    # calculate r. The dimension of r would be L * 1
    con = tf.concat(values=[lstm_hidden_state, coattention_encoding_of_prev_start_word,
                               coattention_encoding_of_prev_end_word], axis=0)
    linear_model = tf.matmul(wd, con, transpose_a=True)
    activated_value = tf.nn.tanh(linear_model)
    tf.reshape(activated_value, [HIDDEN_STATE_SIZE])

    # calculate mt1
    #w1 = tf.get_variable("w1", shape=[POOL_SIZE, HIDDEN_STATE_SIZE, 3 * HIDDEN_STATE_SIZE],
    #                     initializer=weight_initer)
    # b1 = tf.get_variable("b1", shape=[POOL_SIZE, HIDDEN_STATE_SIZE])
    con2 = tf.concat(values=[coattention_encoding_of_word_in_doc, activated_value],axis=0)
    mt1_premax = tf.reshape(tf.matmul(w1, con2), [POOL_SIZE, HIDDEN_STATE_SIZE]) + b1
    mt1_postmax =  tf.reduce_max(mt1_premax, axis=0)
    mt1_postmax_reshaped = tf.reshape(mt1_postmax, [HIDDEN_STATE_SIZE])

    #calculate mt2
    #w2 = tf.get_variable("w2", shape=[POOL_SIZE, HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE],
    #                     initializer=weight_initer)
    # b2 = tf.get_variable("b2", shape=[POOL_SIZE, HIDDEN_STATE_SIZE])
    mt2_premax = tf.reshape(tf.matmul(w2, mt1_postmax_reshaped), [POOL_SIZE, HIDDEN_STATE_SIZE]) + b2
    mt2_postmax =  tf.reduce_max(mt2_premax, axis=0)
    mt2_postmax_reshaped = tf.reshape(mt2_postmax, [HIDDEN_STATE_SIZE])

    #calculate the final HMN output
    #w3 = tf.get_variable("w3", shape=[POOL_SIZE, HIDDEN_STATE_SIZE, 2 * HIDDEN_STATE_SIZE],
    #                     initializer=weight_initer)
    # b3 = tf.get_variable("b3", shape=[POOL_SIZE])
    con3 = tf.concat(values=[mt1_postmax_reshaped, mt2_postmax_reshaped],axis = 0)
    hmn_premax = tf.reshape(tf.matmul(w3, con3), [POOL_SIZE]) + b3
    hmn_postmax = tf.reduce_max(hmn_premax, axis=0)

    return hmn_postmax




