import tensorflow as tf
from keras import layers

# encoder for questions and document
def encoder(question,context,embeddings,hidden_units_size=200):
    batch_size = tf.shape(question)[0]

    # Set up LSTM for encoding the document and question embeddings
    lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_units_size)

    print("get embeddings")
    # Get embedding for question
    q_embedding = tf.nn.embedding_lookup(embeddings,question)
    d_embedding = tf.nn.embedding_lookup(embeddings,context)

    print("got embeddings")

    with tf.variable_scope('document_encoder') as document_scope:
        document_states, _ = layers.RNN(lstm_cell)

    with tf.variable_scope('question_encoder') as question_scope:
        question_states, _ = layers.RNN(lstm_cell)

    print(q_embedding)