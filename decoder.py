# decode the question using the dynamic decoder
import tensorflow as tf
import highway_network as hn

EMBEDDING_SIZE_OF_WORDS = 400
DOCUMENT_SIZE = 632
HIDDEN_STATE_SIZE = 200 # named L in the paper
POOL_SIZE = 16

def decoder(U, s, e, hidden_unit_size=200):
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

    weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    wd_start_word = tf.get_variable("wd_s", shape=[HIDDEN_STATE_SIZE, 5 * HIDDEN_STATE_SIZE],
                                    initializer=weight_initer)
    w1_start_word = tf.get_variable("w1_s", shape=[POOL_SIZE, HIDDEN_STATE_SIZE, 3 * HIDDEN_STATE_SIZE],
                                    initializer=weight_initer)
    w2_start_word = tf.get_variable("w2_s", shape=[POOL_SIZE, HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE],
                                    initializer=weight_initer)
    w3_start_word = tf.get_variable("w3_s", shape=[POOL_SIZE, 1, 2 * HIDDEN_STATE_SIZE],
                                    initializer=weight_initer)
    b1_start_word = tf.get_variable("b1_s", shape=[POOL_SIZE, HIDDEN_STATE_SIZE])
    b2_start_word = tf.get_variable("b2_s", shape=[POOL_SIZE, HIDDEN_STATE_SIZE])
    b3_start_word = tf.get_variable("b3_s", shape=[POOL_SIZE])

    wd_end_word = tf.get_variable("wd_e", shape=[HIDDEN_STATE_SIZE, 5 * HIDDEN_STATE_SIZE],
                                  initializer=weight_initer)
    w1_end_word = tf.get_variable("w1_e", shape=[POOL_SIZE, HIDDEN_STATE_SIZE, 3 * HIDDEN_STATE_SIZE],
                                  initializer=weight_initer)
    w2_end_word = tf.get_variable("w2_e", shape=[POOL_SIZE, HIDDEN_STATE_SIZE, HIDDEN_STATE_SIZE],
                                  initializer=weight_initer)
    w3_end_word = tf.get_variable("w3_e", shape=[POOL_SIZE, 1, 2 * HIDDEN_STATE_SIZE],
                                  initializer=weight_initer)
    b1_end_word = tf.get_variable("b1_e", shape=[POOL_SIZE, HIDDEN_STATE_SIZE])
    b2_end_word = tf.get_variable("b2_e", shape=[POOL_SIZE, HIDDEN_STATE_SIZE])
    b3_end_word = tf.get_variable("b3_e", shape=[POOL_SIZE])


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
        s, s_logits = hn.highway_network(U, hi, u_s, u_e, wd_start_word, w1_start_word, w2_start_word, w3_start_word,
                    b1_start_word, b2_start_word, b3_start_word)
        print("s shape:",s.shape)
        print("s.dtype = ",s.dtype)
        u_s = tf.gather_nd(params=tf.transpose(U, perm=[0 , 2, 1]),
                           indices=tf.stack([tf.range(batch_size,dtype=tf.int32),tf.reshape(s, shape=[s.shape[0]])], axis=1))
        print("IM HERE u_s.shape: ",u_s.shape)
        print("u_s.dtype ",u_s.dtype)

        # e is the end index
        e, e_logits = hn.highway_network(U, hi, u_s, u_e, wd_end_word, w1_end_word, w2_end_word, w3_end_word,
                    b1_end_word, b2_end_word, b3_end_word)
        print("e.dtype = ",e.dtype)

        u_e = tf.gather_nd(params=tf.transpose(U, perm=[0, 2, 1]),
                         indices=tf.stack([tf.range(batch_size, dtype=tf.int32), tf.reshape(e, shape=[e.shape[0]])], axis=1))

        print("u_e.shape ",u_e.shape)
        print("u_e.dtype :",u_e.dtype)
        hi,ch = lstm_cell(inputs=tf.concat([u_s, u_e],axis=1), state=ch)

    return s,e

print()
U = tf.placeholder(shape=[10, 400, 632], dtype = tf.float32)
s = tf.placeholder(shape=[10], dtype = tf.int32)
e = tf.placeholder(shape=[10], dtype = tf.int32)
decoder(U, s, s)

