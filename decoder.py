# decode the question using the dynamic decoder
import tensorflow as tf
import highway_network as hn

def decoder(U, s, e, hidden_unit_size = 200, pool_size = 16):
    """
    :param U: This is output of the encoder
    :param batch_size:
    :param s_init:
    :param e_init:
    :return:
    """
    batch_size = U.shape[0]
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_unit_size, dtype = tf.float32)
    ch = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    hi, _ = ch

    # Initialise variables to load them into the default 
    weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    with tf.variable_scope('start_word') as scope1:
        wd = tf.get_variable("wd", shape=[hidden_unit_size, 5 * hidden_unit_size],
                                        initializer=weight_initer)
        w1 = tf.get_variable("w1", shape=[pool_size, hidden_unit_size, 3 * hidden_unit_size],
                                        initializer=weight_initer)
        w2 = tf.get_variable("w2", shape=[pool_size, hidden_unit_size, hidden_unit_size],
                                        initializer=weight_initer)
        w3 = tf.get_variable("w3", shape=[pool_size, 1, 2 * hidden_unit_size],
                                        initializer=weight_initer)
        b1 = tf.get_variable("b1", shape=[pool_size, hidden_unit_size])
        b2 = tf.get_variable("b2", shape=[pool_size, hidden_unit_size])
        b3 = tf.get_variable("b3", shape=[pool_size])

    with tf.variable_scope('end_word') as scope2:
        wd = tf.get_variable("wd", shape=[hidden_unit_size, 5 * hidden_unit_size],
                                        initializer=weight_initer)
        w1 = tf.get_variable("w1", shape=[pool_size, hidden_unit_size, 3 * hidden_unit_size],
                                        initializer=weight_initer)
        w2 = tf.get_variable("w2", shape=[pool_size, hidden_unit_size, hidden_unit_size],
                                        initializer=weight_initer)
        w3 = tf.get_variable("w3", shape=[pool_size, 1, 2 * hidden_unit_size],
                                        initializer=weight_initer)
        b1 = tf.get_variable("b1", shape=[pool_size, hidden_unit_size])
        b2 = tf.get_variable("b2", shape=[pool_size, hidden_unit_size])
        b3 = tf.get_variable("b3", shape=[pool_size])


    u_s = tf.gather_nd(params=tf.transpose(U, perm=[0, 2, 1]),
                       indices=tf.stack([tf.range(batch_size, dtype=tf.int32), s], axis=1))
    print("u_s shape: ",u_s.shape)
    # 10 * 400
    u_e = tf.gather_nd(params=tf.transpose(U, perm=[0, 2, 1]),
                       indices=tf.stack([tf.range(batch_size, dtype=tf.int32), e], axis=1))
    print("u_e shape: ",u_e.shape)

    print("HERE")
    for i in range(4):
        # s is start index
        with tf.variable_scope('start_word', reuse=True) as scope1:
            s, s_logits = hn.highway_network(U, hi, u_s, u_e, hidden_unit_size = hidden_unit_size, pool_size = pool_size)
            print("s shape:",s.shape)
            print("s.dtype = ",s.dtype)
            u_s = tf.gather_nd(params=tf.transpose(U, perm=[0 , 2, 1]),
                            indices=tf.stack([tf.range(batch_size,dtype=tf.int32),tf.reshape(s, shape=[s.shape[0]])], axis=1))
            print("IM HERE u_s.shape: ",u_s.shape)
            print("u_s.dtype ",u_s.dtype)

        # e is the end index
        with tf.variable_scope('end_word', reuse=True) as scope2:
            e, e_logits = hn.highway_network(U, hi, u_s, u_e, hidden_unit_size = hidden_unit_size, pool_size = pool_size)
            print("e.dtype = ",e.dtype)
            u_e = tf.gather_nd(params=tf.transpose(U, perm=[0, 2, 1]),
                            indices=tf.stack([tf.range(batch_size, dtype=tf.int32), tf.reshape(e, shape=[e.shape[0]])], axis=1))
            print("u_e.shape ",u_e.shape)
            print("u_e.dtype :",u_e.dtype)

        hi,ch = lstm_cell(inputs=tf.concat([u_s, u_e],axis=1), state=ch)

    return s,e, s_logits,e_logits

print()
U = tf.placeholder(shape=[10, 400, 632], dtype = tf.float32)
s = tf.placeholder(shape=[10], dtype = tf.int32)
e = tf.placeholder(shape=[10], dtype = tf.int32)
decoder(U, s, s)

