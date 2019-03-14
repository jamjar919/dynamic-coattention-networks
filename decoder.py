# decode the question using the dynamic decoder
import tensorflow as tf
import highway_network as hn
import numpy as np

def decoder(U, hidden_unit_size = 200, pool_size = 16):
    """
    :param U: This is output of the encoder
    :param batch_size:
    :param s_init:
    :param e_init:
    :return:
    """
    batch_size = U.shape[0]
    iterations = 4

    sv = tf.random_uniform(tf.TensorShape([batch_size]), minval=0, maxval=U.shape[2], dtype=tf.int32)
    ev = tf.random_uniform(tf.TensorShape([batch_size]), minval=0, maxval=U.shape[2] + 1, dtype=tf.int32)

    print(sv)
    print(ev)

    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units = hidden_unit_size, dtype = tf.float32)
    ch = lstm_cell.zero_state(batch_size, dtype=tf.float32) # Return 0 state filled tensor.
    hi, _ = ch
    print("hi.shape", hi.shape) # 10x200

    # Initialise variables to load them into the default 
    weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    # Weights and biases for the HMN that will calculate a start index. 
    
    with tf.variable_scope('start_word') as scope1:
        # wd dim: lx5l
        wd = tf.get_variable("wd", shape=[hidden_unit_size, 5 * hidden_unit_size],
                                        initializer=weight_initer)
        # w1 dim: pxlx3l
        w1 = tf.get_variable("w1", shape=[pool_size, hidden_unit_size, 3 * hidden_unit_size],
                                        initializer=weight_initer)
        # w2 dim: pxlxl
        w2 = tf.get_variable("w2", shape=[pool_size, hidden_unit_size, hidden_unit_size],
                                        initializer=weight_initer)
        #w3 dim: px1x2l
        w3 = tf.get_variable("w3", shape=[pool_size, 1, 2 * hidden_unit_size],
                                        initializer=weight_initer)
        b1 = tf.get_variable("b1", shape=[pool_size, hidden_unit_size, ]) # b1 dim: pxl
        b2 = tf.get_variable("b2", shape=[pool_size, hidden_unit_size, ]) # b2 dim: pxl
        b3 = tf.get_variable("b3", shape=[pool_size]) #b3 dim: px1

    with tf.variable_scope('end_word') as scope2:
        wd = tf.get_variable("wd", shape=[hidden_unit_size, 5 * hidden_unit_size],
                                        initializer=weight_initer)
        w1 = tf.get_variable("w1", shape=[pool_size, hidden_unit_size, 3 * hidden_unit_size],
                                        initializer=weight_initer)
        w2 = tf.get_variable("w2", shape=[pool_size, hidden_unit_size, hidden_unit_size],
                                        initializer=weight_initer)
        w3 = tf.get_variable("w3", shape=[pool_size, 1,  2 * hidden_unit_size],
                                        initializer=weight_initer)
        b1 = tf.get_variable("b1", shape=[pool_size, hidden_unit_size, ])
        b2 = tf.get_variable("b2", shape=[pool_size, hidden_unit_size, ])
        b3 = tf.get_variable("b3", shape=[pool_size, 1])
    
    for i in range(iterations):
        # s is start index
        u_s = tf.gather_nd(params=U,indices=tf.stack([tf.range(batch_size,dtype=tf.int32),sv],axis=1))
        u_e = tf.gather_nd(params=U,indices=tf.stack([tf.range(batch_size,dtype=tf.int32),ev],axis=1))
        usue = tf.concat([u_s,u_e],axis=1)
        print("usue shape", usue.shape)
        with tf.variable_scope('start_word', reuse = True) as scope1:
            # Returns argmax  as well as all outputs of the highway network α1,...,α_m   (equation (6))
            sv, s_logits = hn.highway_network(U, hi, u_s, u_e, hidden_unit_size = hidden_unit_size, pool_size = pool_size)

        # e is the end index
        with tf.variable_scope('end_word', reuse = True) as scope2:
            ev, e_logits = hn.highway_network(U, hi, u_s, u_e, hidden_unit_size = hidden_unit_size, pool_size = pool_size)

        hi,ch = lstm_cell(inputs=usue, state=ch) # 

        #hi = tf.Print(hi,[],"ITERATION") # Print just the message. 

    return sv, ev, s_logits, e_logits

if __name__ == "__main__":
    print("Running decoder by itself for debug purposes.")
    U = tf.placeholder(shape=[10, 632, 400], dtype = tf.float32)
    decoder(U)
