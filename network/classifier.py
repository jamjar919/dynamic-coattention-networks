import tensorflow as tf
from network.encoder import encoder
import numpy as np
from network.config import CONFIG

def get_batch(batch, batch_size = CONFIG.BATCH_SIZE, max_length_question = CONFIG.MAX_QUESTION_LENGTH, max_length_context = CONFIG.MAX_CONTEXT_LENGTH):
    question_batch = np.array(list(map(lambda qas: (qas["question"]), batch))).reshape(batch_size, max_length_question)
    context_batch = np.array(list(map(lambda qas: (qas["context"]), batch))).reshape(batch_size, max_length_context)
    has_answer_batch = np.array(list(map(lambda qas: (qas["has_answer"]), batch))).reshape(batch_size)
    return question_batch, context_batch, has_answer_batch

def get_feed_dict(question_batch, context_batch, answer_batch, dropout_keep_prob, index2embedding) :
    graph = tf.get_default_graph()
    return {
                graph.get_tensor_by_name("question_batch_ph:0") : question_batch,
                graph.get_tensor_by_name("context_batch_ph:0") : context_batch,
                graph.get_tensor_by_name("has_answer_ph:0"): answer_batch,
                graph.get_tensor_by_name("dropout_keep_ph:0") : dropout_keep_prob,
                graph.get_tensor_by_name("embedding_ph:0"): index2embedding
            }

def build_classifier(embedding):
    dropout_keep_rate = tf.placeholder(dtype = tf.float32, name = "dropout_keep_ph")
    question_batch_ph = tf.placeholder(dtype=tf.int32, shape = [CONFIG.BATCH_SIZE, CONFIG.MAX_QUESTION_LENGTH], name='question_batch_ph')
    context_batch_ph = tf.placeholder(dtype=tf.int32, shape = [CONFIG.BATCH_SIZE, CONFIG.MAX_CONTEXT_LENGTH], name='context_batch_ph')
    # Word index placeholders
    has_answer_ph = tf.placeholder(dtype=tf.float32,shape=[CONFIG.BATCH_SIZE,], name='has_answer_ph')

    #U = tf.placeholder(dtype = tf.float32, shape = [64,633,400])
    U, _ = encoder(question_batch_ph,context_batch_ph, embedding, dropout_keep_rate)
    U = U[:,0:410,:]
    print("U shape", U.shape)
    initer = tf.contrib.layers.xavier_initializer()
    LAYER_SIZE = 20
    W1 = tf.get_variable("W1", shape = [U.shape[2],LAYER_SIZE], initializer = initer, dtype = tf.float32)
    b1 = tf.get_variable("b1", shape = [U.shape[1],LAYER_SIZE], initializer = initer, dtype = tf.float32)
    W2 = tf.get_variable("W2", shape = [LAYER_SIZE,1], initializer = initer, dtype = tf.float32)
    #W2 = tf.Print(W2, [W2], "W2 values", summarize = 10)
    b2 = tf.get_variable("b2", shape = [U.shape[1],1], initializer = initer, dtype = tf.float32)
    W3 = tf.get_variable("W3", shape = [U.shape[1],1], initializer = initer, dtype = tf.float32)
    b3 = tf.get_variable("b3", shape = [1,1], initializer = initer, dtype = tf.float32)

    batch_size = U.shape[0]
    W1_batch = tf.stack([W1] * batch_size); b1_batch = tf.stack([b1] * batch_size)
    W2_batch = tf.stack([W2] * batch_size); b2_batch = tf.stack([b2] * batch_size)
    W3_batch = tf.stack([W3] * batch_size); b3_batch = tf.stack([b3] * batch_size)
    out = tf.nn.tanh(tf.matmul(U,W1_batch) + b1_batch)
    out = tf.nn.dropout(out, keep_prob = dropout_keep_rate)
    print("after first matmul: ", out.shape)
    out = tf.nn.tanh(tf.matmul(out, W2_batch)+b2_batch)
    out = tf.nn.dropout(out, keep_prob = dropout_keep_rate)
    print("after second matmul: ", out.shape)
    out = tf.matmul(tf.transpose(out, perm = [0,2,1]), W3_batch) + b3_batch
    print("OUT.shape = ",out.shape)
    out = tf.squeeze(out)
    out_2 = tf.sigmoid(out, name = "classifier_output")
    print("output shape from classifier: ", out.shape)
    print("output shape 2 from classifier: ", out_2.shape)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = has_answer_ph, logits = out)
    loss = tf.identity(loss, name = "loss_v2_classifier")
    #loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = has_answer_ph, logits = out, name = "loss_v2_classifier")
    optimizer = tf.train.AdamOptimizer(CONFIG.LEARNING_RATE)
    train_op = optimizer.minimize(loss, name = "train_op_classifier")

    return train_op, loss, out_2


if __name__ == "__main__":
    print("Running classifier by itself for debug purposes.")
    
    embedding = tf.placeholder(shape=[1542, 300],
                               dtype=tf.float32, name='embedding')
 

    build_classifier(embedding)