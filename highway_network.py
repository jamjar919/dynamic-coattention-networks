import tensorflow as tf


HIDDEN_STATE_SIZE = 200 # named L in the paper
POOL_SIZE = 16

def transpose(tensor):
    return tf.transpose(tensor,perm=[0,2,1])

def highway_network(U, lstm_hidden_state,
                    coattention_encoding_of_prev_start_word,
                    coattention_encoding_of_prev_end_word,
                    wd, w1, w2, w3,
                    b1, b2, b3):

    U_transpose = tf.transpose(U,perm=[1, 2, 0])
    print("u_t shape:", U_transpose.shape)
    fn = lambda batch_of_word_encodings : highway_network_batch(batch_of_word_encodings, lstm_hidden_state,
                    coattention_encoding_of_prev_start_word,
                    coattention_encoding_of_prev_end_word,
                    wd, w1, w2, w3,
                    b1, b2, b3)
    # returns 10 * 1
    return tf.map_fn(fn, U_transpose)

# U_transpose is of size 632 * 400
# def highway_network_matrix(U_transpose, lstm_hidden_state,
#                     coattention_encoding_of_prev_start_word,
#                     coattention_encoding_of_prev_end_word,
#                     wd, w1, w2, w3,
#                     b1, b2, b3):
#
#     #U_transpose = tf.transpose(U, perm=[1, 0])
#     fn = lambda col : highway_network_single(col, lstm_hidden_state,
#                     coattention_encoding_of_prev_start_word,
#                     coattention_encoding_of_prev_end_word,
#                     wd, w1, w2, w3,
#                     b1, b2, b3)
#     # returns 1 number
#     return tf.reduce_max(tf.map_fn(fn, U_transpose))

def highway_network_batch(batch_of_word_encodings,
                    lstm_hidden_state,
                    coattention_encoding_of_prev_start_word,
                    coattention_encoding_of_prev_end_word,
                    wd, w1, w2, w3,
                    b1, b2, b3):
    # weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    #wd = tf.get_variable("wd", shape=[HIDDEN_STATE_SIZE, 5 * HIDDEN_STATE_SIZE],
    #                     initializer=weight_initer)

    # calculate r. The dimension of r would be l * 1
    batch_size = batch_of_word_encodings.shape[0]
    print("batch_size : ",batch_size)
    con = tf.concat(values=[lstm_hidden_state, coattention_encoding_of_prev_start_word,
                               coattention_encoding_of_prev_end_word], axis=1)
    con = tf.reshape(con,[con.shape[0],con.shape[1],1])
    print("con.shape:", con.shape)
    # wd = tf.tile(wd,[batch_size])
    print("wd.shape: ",wd.shape)
    linear_model = tf.map_fn(lambda x: tf.matmul(wd,x),con)
    print("linear_model shape: ",linear_model.shape)
    activated_value = tf.nn.tanh(linear_model)
    tf.reshape(activated_value, [HIDDEN_STATE_SIZE])

    # calculate mt1
    #w1 = tf.get_variable("w1", shape=[POOL_SIZE, HIDDEN_STATE_SIZE, 3 * HIDDEN_STATE_SIZE],
    #                     initializer=weight_initer)
    # b1 = tf.get_variable("b1", shape=[POOL_SIZE, HIDDEN_STATE_SIZE])
    con2 = tf.concat(values=[coattention_encoding_of_word_in_doc, activated_value])
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
    con3 = tf.concat(values=[mt1_postmax_reshaped, mt2_postmax_reshaped])
    hmn_premax = tf.reshape(tf.matmul(w3, con3), [POOL_SIZE]) + b3
    hmn_postmax = tf.reduce_max(hmn_premax, axis=0)

    return hmn_postmax