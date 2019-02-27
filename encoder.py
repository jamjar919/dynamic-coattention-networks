import tensorflow as tf
from keras import layers

def encoder(questions,contexts,embedding,hidden_units_size=200):
    '''
        Build the model for the document encoder
        questions: Tensor of questions
        contexts: Tensor of contexts
        embedding: Mappings from encoded questions to GLoVE vectors
    '''
    batch_size = questions.get_shape()[0]

    print("Batch size", batch_size)
    print("Shape of questions", questions.get_shape())
    print("Shape of contexts", contexts.get_shape())

    with tf.variable_scope('embedding') as scope:
        # Vectorise the contexts and questions
        context_vector = tf.map_fn(lambda x:  tf.nn.embedding_lookup(embedding, x), contexts, dtype=tf.float32)
        question_vector = tf.map_fn(lambda x:  tf.nn.embedding_lookup(embedding, x), questions, dtype=tf.float32)
        print(question_vector)

        lstm_enc = tf.nn.rnn_cell.LSTMCell(hidden_units_size)
