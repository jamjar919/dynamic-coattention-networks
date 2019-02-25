import tensorflow as tf

# encoder for questions and document
def encoder(question,context,embeddings,hidden_units_size=200):
    # Set up LSTM for encoding the document and question embeddings
    lstm_cell = tf.contrib.rnn_cell.LSTMCell(hidden_units_size)
    pass;
