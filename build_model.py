import tensorflow as tf
import numpy as np
from encoder import encoder
from decoder import decoder
from config import CONFIG

def build_model(embedding):
    dropout_keep_rate = tf.placeholder(dtype = tf.float32, name = "dropout_keep_ph")
    batch_size_ph = tf.placeholder(dtype = tf.int32, name = "batch_size_ph")
    question_batch_placeholder = tf.placeholder(dtype=tf.int32, shape = [CONFIG.BATCH_SIZE, CONFIG.MAX_QUESTION_LENGTH], name='question_batch_ph')
    context_batch_placeholder = tf.placeholder(dtype=tf.int32, shape = [CONFIG.BATCH_SIZE, CONFIG.MAX_CONTEXT_LENGTH], name='context_batch_ph')
    # Create encoder. (Encoder will also return the sequence length of the context (i.e. how much of each batch element is unpadded))
    U = encoder(question_batch_placeholder,context_batch_placeholder, dropout_keep_rate, embedding, hidden_unit_size=CONFIG.HIDDEN_UNIT_SIZE, embedding_vector_size=CONFIG.EMBEDDING_DIMENSION)
    # Word index placeholders
    answer_start = tf.placeholder(dtype=tf.int32,shape=[None], name='answer_start_true_ph')
    answer_end = tf.placeholder(dtype=tf.int32,shape=[None], name='answer_end_true_ph')

    # Create decoder 
    s, e, alphas, betas = decoder(U, hidden_unit_size=CONFIG.HIDDEN_UNIT_SIZE, pool_size=CONFIG.POOL_SIZE) # Pass also the seq_length from encoder and max_length.

    s = tf.identity(s, name='answer_start')
    e = tf.identity(e, name='answer_end')

    losses_alpha = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_start, logits=a) for a in alphas]
    losses_alpha = [tf.reduce_mean(x) for x in losses_alpha]
    losses_beta = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_end, logits=b) for b in betas]
    losses_beta = [tf.reduce_mean(x) for x in losses_beta]
    loss = tf.reduce_sum([losses_alpha, losses_beta])
    loss = tf.identity(loss, name = "loss")

    original_optimizer = tf.train.AdamOptimizer(CONFIG.LEARNING_RATE)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer, clip_norm=CONFIG.CLIP_NORM)
    train_op = optimizer.minimize(loss)

    return train_op, loss, s , e


def get_feed_dict(question_batch, context_batch, answer_start_batch, answer_end_batch, dropout_keep_prob, index2embedding) :
    graph = tf.get_default_graph()
    return {
                graph.get_tensor_by_name("question_batch_ph:0") : question_batch,
                graph.get_tensor_by_name("context_batch_ph:0") : context_batch,
                graph.get_tensor_by_name("answer_start_true_ph:0"): answer_start_batch,
                graph.get_tensor_by_name("answer_end_true_ph:0") : answer_end_batch,
                graph.get_tensor_by_name("dropout_keep_ph:0") : dropout_keep_prob,
                graph.get_tensor_by_name("embedding_ph:0"): index2embedding
            }