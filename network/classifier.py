import tensorflow as tf
from network.encoder import encoder
import numpy as np
from network.config import CONFIG

def get_batch(batch, batch_size = CONFIG.BATCH_SIZE, max_length_question = CONFIG.MAX_QUESTION_LENGTH, max_length_context = CONFIG.MAX_CONTEXT_LENGTH):
    question_batch = np.array(list(map(lambda qas: (qas["question"]), batch))).reshape(batch_size, max_length_question)
    context_batch = np.array(list(map(lambda qas: (qas["context"]), batch))).reshape(batch_size, max_length_context)
    has_answer_batch = np.array(list(map(lambda qas: (qas["has_answer"]), batch))).reshape(batch_size)
    return question_batch, context_batch, has_answer_batch

def build_classifier(embedding):
    dropout_keep_rate = tf.placeholder(dtype = tf.float32, name = "dropout_keep_ph")
    batch_size_ph = tf.placeholder(dtype = tf.int32, name = "batch_size_ph")
    question_batch_placeholder = tf.placeholder(dtype=tf.int32, shape = [CONFIG.BATCH_SIZE, CONFIG.MAX_QUESTION_LENGTH], name='question_batch_ph')
    context_batch_placeholder = tf.placeholder(dtype=tf.int32, shape = [CONFIG.BATCH_SIZE, CONFIG.MAX_CONTEXT_LENGTH], name='context_batch_ph')
    # Word index placeholders
    answer_start = tf.placeholder(dtype=tf.int32,shape=[None], name='answer_start_true_ph')
    answer_end = tf.placeholder(dtype=tf.int32,shape=[None], name='answer_end_true_ph')
    max_context_length = CONFIG.MAX_CONTEXT_LENGTH

    # Create encoder. (Encoder will also return the sequence length of the context (i.e. how much of each batch element is unpadded))
    U, context_seq_length = encoder(question_batch_placeholder,context_batch_placeholder, embedding, dropout_keep_rate)
    initer = tf.contrib.layers.xavier_initializer()
    LAYER_SIZE = 10
    W1 = tf.get_variable("W1", shape = [U.shape[2],LAYER_SIZE], initializer = initer, dtype = tf.float32)
    b1 = tf.get_variable("b1", shape = [U.shape[1],LAYER_SIZE], initializer = initer, dtype = tf.float32)
    W2 = tf.get_variable("W2", shape = [LAYER_SIZE,1], initializer = initer, dtype = tf.float32)
    b2 = tf.get_variable("b2", shape = [U.shape[1],1], initializer = initer, dtype = tf.float32)
    W3 = tf.get_variable("W3", shape = [1,U.shape[1]], initializer = initer, dtype = tf.float32)
    b3 = tf.get_variable("b3", shape = [1,1], initializer = initer, dtype = tf.float32)

    batch_size = U.shape[0]
    W1_batch = tf.stack([W1] * batch_size); b1_batch = tf.stack([b1] * batch_size)
    W2_batch = tf.stack([W2] * batch_size); b2_batch = tf.stack([b2] * batch_size)
    W3_batch = tf.stack([W3] * batch_size); b3_batch = tf.stack([b3] * batch_size)
    out = tf.tanh(tf.matmul(U,W1_batch)+b1_batch)
    out = tf.tanh(tf.matmul(out,W2_batch)+b2_batch)
    out = tf.tanh(tf.matmul(out, W3_batch) + b3_batch)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels = answer_start, logits = out)
    optimizer = tf.train.AdamOptimizer(CONFIG.LEARNING_RATE)
    train_op = optimizer.minimize(loss, name = "train_op")
