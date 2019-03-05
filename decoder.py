# decode the question using the dynamic decoder
import tensorflow as tf
import random

HIDDEN_STATE_SIZE = 200
EMBEDDING_SIZE_OF_WORDS = 400
DOCUMENT_SIZE = 632
def decoder(encoder_states, batch_size, s_init, e_init):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_STATE_SIZE)
    state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    for i in range(4):
        


    pass;
