import tensorflow as tf

# encoder for questions and document
def encoder(question,context,embeddings,hidden_units_size=200):

    # Set up LSTM for encoding the document and question embeddings
    print("lstm cell")
    lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_units_size)
    print(question)

    # Get embedding for question
    q_embedding = tf.nn.embedding_lookup(embeddings,question)
    d_embedding = tf.nn.embedding_lookup(embeddings,context)

    print(q_embedding)