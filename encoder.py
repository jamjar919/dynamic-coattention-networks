import tensorflow as tf
from keras import layers

def encoder(questions,contexts,embeddings,hidden_units_size=200):
    '''
        Build the model for the document encoder
        questions: Tensor of questions
        contexts: Tensor of contexts
        embeddings: Mappings from encoded questions to GLoVE vectors
    '''
    batch_size = tf.shape(questions)[0]

    with tf.variable_scope('embedding') as scope:
        # Vectorise the contexts and questions
        context_vector = tf.map_fn(lambda x: tf.nn.embedding_lookup(embedding, x), contexts)
        question_vector = tf.map_fn(lambda x: tf.nn.embedding_lookup(embedding, x), questions)

    print(q_embedding)