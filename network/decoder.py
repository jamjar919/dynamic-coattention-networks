# decode the question using the dynamic decoder
import tensorflow as tf
from network.highway_network import highway_network
import numpy as np
from network.config import CONFIG

def decoder(U, context_seq_length, max_context_length, hidden_unit_size = CONFIG.HIDDEN_UNIT_SIZE, pool_size = CONFIG.POOL_SIZE):
    """
    :param U: This is output of the encoder
    :param batch_size:
    :param s_init:
    :param e_init:
    :return:
    """

    batch_size = U.shape[0]
    iterations = 4


    initer = tf.contrib.layers.xavier_initializer()     
    with tf.variable_scope('HMN_start'):
        # wd dim: lx5l
        Wd = tf.get_variable("Wd", shape=[hidden_unit_size, 5 * hidden_unit_size],
                                        initializer=initer)
        # w1 dim: pxlx3l
        W1 = tf.get_variable("W1", shape=[pool_size, hidden_unit_size, 3 * hidden_unit_size],
                                        initializer=initer)
        # w2 dim: pxlxl
        W2 = tf.get_variable("W2", shape=[pool_size, hidden_unit_size, hidden_unit_size],
                                        initializer=initer)
        #w3 dim: px1x2l
        W3 = tf.get_variable("W3", shape=[pool_size, 1, 2 * hidden_unit_size],
                                        initializer=initer)
        b1 = tf.get_variable("b1", shape=[pool_size, hidden_unit_size, ], initializer = tf.zeros_initializer()) # b1 dim: pxl
        b2 = tf.get_variable("b2", shape=[pool_size, hidden_unit_size, ], initializer = tf.zeros_initializer()) # b2 dim: pxl
        b3 = tf.get_variable("b3", shape=[pool_size,1], initializer=tf.zeros_initializer()) #b3 dim: px1

    with tf.variable_scope('HMN_end'):
        Wd = tf.get_variable("Wd", shape=[hidden_unit_size, 5 * hidden_unit_size],
                                        initializer=initer)
        # w1 dim: pxlx3l
        W1 = tf.get_variable("W1", shape=[pool_size, hidden_unit_size, 3 * hidden_unit_size],
                                        initializer=initer)
        # w2 dim: pxlxl
        W2 = tf.get_variable("W2", shape=[pool_size, hidden_unit_size, hidden_unit_size],
                                        initializer=initer)
        #w3 dim: px1x2l
        W3 = tf.get_variable("W3", shape=[pool_size, 1, 2 * hidden_unit_size],
                                        initializer=initer)
        b1 = tf.get_variable("b1", shape=[pool_size, hidden_unit_size, ], initializer = tf.zeros_initializer()) # b1 dim: pxl
        b2 = tf.get_variable("b2", shape=[pool_size, hidden_unit_size, ], initializer = tf.zeros_initializer()) # b2 dim: pxl
        b3 = tf.get_variable("b3", shape=[pool_size,1], initializer=tf.zeros_initializer()) #b3 dim: px1


    s = tf.zeros(tf.TensorShape([batch_size]), dtype=tf.int32)
    e = tf.zeros(tf.TensorShape([batch_size]), dtype=tf.int32)

    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units = hidden_unit_size, dtype = tf.float32)
    lstm_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) # Return 0 state filled tensor.
    h_i, _ = lstm_state
    print("h_i.shape", h_i.shape) # 10x200
   
    alphas, betas = [] , []
    for i in range(iterations):
        # s is start index
        u_s = tf.gather_nd(params=U,indices=tf.stack([tf.range(batch_size,dtype=tf.int32),s],axis=1))
        u_e = tf.gather_nd(params=U,indices=tf.stack([tf.range(batch_size,dtype=tf.int32),e],axis=1))
        us_ue_concat = tf.concat([u_s,u_e],axis=1)
        h_i,lstm_state = lstm_cell(inputs=us_ue_concat, state=lstm_state) # 
        print("usue concat shape", us_ue_concat.shape)
        with tf.variable_scope('HMN_start', reuse = True):
            # Returns argmax  as well as all outputs of the highway network α1,...,α_m   (equation (6))
            s, s_logits = highway_network(U, h_i, u_s, u_e, context_seq_length, max_context_length, hidden_unit_size = hidden_unit_size, pool_size = pool_size)
            alphas.append(s_logits)
        with tf.variable_scope('HMN_end', reuse = True):
            e, e_logits = highway_network(U, h_i, u_s, u_e, context_seq_length, max_context_length, hidden_unit_size = hidden_unit_size, pool_size = pool_size)
            betas.append(e_logits)

    return s, e, alphas , betas

if __name__ == "__main__":
    print("Running decoder by itself for debug purposes.")
    U = tf.placeholder(shape=[16, 632, 400], dtype = tf.float32)
    seq_length = tf.placeholder(shape =  [16,], dtype = tf.int32)
    max_length = 632
    decoder(U, seq_length, max_length)
