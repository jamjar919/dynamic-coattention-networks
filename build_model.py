import tensorflow as tf
import numpy as np
from encoder import encoder
from decoder import decoder
from config import CONFIG

def get_batch(batch, batch_size = CONFIG.BATCH_SIZE, max_length_question = CONFIG.MAX_QUESTION_LENGTH, max_length_context = CONFIG.MAX_CONTEXT_LENGTH):
    question_batch = np.array(list(map(lambda qas: (qas["question"]), batch))).reshape(batch_size, max_length_question)
    context_batch = np.array(list(map(lambda qas: (qas["context"]), batch))).reshape(batch_size, max_length_context)
    answer_start_batch = np.array(list(map(lambda qas: (qas["answer_start"]), batch))).reshape(batch_size)
    answer_end_batch = np.array(list(map(lambda qas: (qas["answer_end"]), batch))).reshape(batch_size)
    return question_batch, context_batch, answer_start_batch, answer_end_batch

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
    alphas = tf.identity(alphas, name='alphas')
    betas = tf.identity(betas,name ='betas')
    print("alphas.shape = ",alphas.shape)
    losses_alpha = tf.map_fn(lambda a : tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_start, logits=a), alphas) 
    print("losses_alpha.shape = ",losses_alpha.shape)
    losses_alpha = tf.reshape(tf.reduce_mean(losses_alpha,axis = 1), shape = [-1,1])
    print("losses_alpha.shape = ",losses_alpha.shape)

    losses_beta = tf.map_fn(lambda b : tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_end, logits=b), betas )
    print("losses_beta.shape = ",losses_beta.shape)
    losses_beta = tf.reshape(tf.reduce_mean(losses_beta,axis = 1), shape = [-1,1])
    print("losses_beta.shape = ", losses_beta.shape)

    # losses_alpha = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_start, logits=a) for a in alphas]
    # losses_alpha = [tf.reduce_mean(x) for x in losses_alpha]
    # losses_beta = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_end, logits=b) for b in betas]
    # losses_beta = [tf.reduce_mean(x) for x in losses_beta]
    
    losses = tf.concat([losses_alpha,losses_beta],axis=0)
    print("losses.shape = ",losses.shape) # should be 8 losses: 4 from alphas and 4 from betas
    loss = tf.reduce_sum(losses, axis = 0,  name = "loss_to_optimize")  # calculate the mean of 8 values
    print("loss.shape = ", loss.shape)

    original_optimizer = tf.train.AdamOptimizer(CONFIG.LEARNING_RATE)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer, clip_norm=CONFIG.CLIP_NORM)
    train_op = optimizer.minimize(loss, name = "train_op")

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