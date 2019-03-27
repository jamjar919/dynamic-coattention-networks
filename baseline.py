from config import CONFIG
import numpy as np
import tensorflow as tf
from dataset import Dataset
import sys

D = Dataset(CONFIG.EMBEDDING_FILE)
index2embedding = D.index2embedding
padded_data, (max_length_question, max_length_context) = D.load_questions(CONFIG.QUESTION_FILE)

print("Loaded data")

# Train now
tf.reset_default_graph()

embedding = tf.placeholder(shape = [len(index2embedding), CONFIG.EMBEDDING_DIMENSION], dtype=tf.float32, name='embedding')
question_batch_placeholder = tf.placeholder(dtype=tf.int32, shape = [CONFIG.BATCH_SIZE, max_length_question], name='question_batch')
context_batch_placeholder = tf.placeholder(dtype=tf.int32, shape = [CONFIG.BATCH_SIZE, max_length_context], name='context_batch')
q_mask_placeholder = tf.placeholder(dtype=tf.bool, shape=(None, max_length_question),
    name="q_mask_placeholder")
c_mask_placeholder = tf.placeholder(dtype=tf.bool, shape=(None, max_length_context),
    name="c_mask_placeholder")
labels_placeholderS = tf.placeholder(tf.int32, (CONFIG.BATCH_SIZE), name="label_phS")
labels_placeholderE = tf.placeholder(tf.int32, (CONFIG.BATCH_SIZE), name="label_phE")

embedded_q = tf.nn.embedding_lookup(params=embedding, ids=question_batch_placeholder)
embedded_c = tf.nn.embedding_lookup(params=embedding, ids=context_batch_placeholder)

with tf.variable_scope("rnn", reuse=None):
    cell = tf.contrib.rnn.GRUCell(200)
    q_sequence_length = tf.reduce_sum(tf.cast(q_mask_placeholder, tf.int32), axis=1)
    q_sequence_length = tf.reshape(q_sequence_length, [-1, ])
    c_sequence_length = tf.reduce_sum(tf.cast(c_mask_placeholder, tf.int32), axis=1)
    c_sequence_length = tf.reshape(c_sequence_length, [-1, ])

    q_outputs, q_final_state = tf.nn.dynamic_rnn(cell=cell, inputs=embedded_q,
        sequence_length=q_sequence_length, dtype=tf.float32,
        time_major=False)

    question_rep = q_final_state

with tf.variable_scope("rnn", reuse=True):
    c_outputs, c_final_state = tf.nn.dynamic_rnn(cell=cell, inputs=embedded_c,
        sequence_length=c_sequence_length,
        initial_state=question_rep,
        time_major=False)

attention = tf.einsum('ik,ijk->ij', question_rep, c_outputs)
float_mask = tf.cast(c_mask_placeholder, dtype=tf.float32)
knowledge_vector = attention * float_mask

xe = tf.contrib.keras.layers.Dense(max_length_context, activation='linear')(knowledge_vector)
xs = tf.contrib.keras.layers.Dense(max_length_context, activation='linear')(knowledge_vector)

xs = xs * float_mask
xe = xe * float_mask

cross_entropyS = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholderS, logits=xs,
                                                            name="cross_entropyS")
cross_entropyE = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholderE, logits=xe,
                                                            name="cross_entropyE")
predictionS = tf.argmax(xs, 1)
predictionE = tf.argmax(xe, 1)

loss = tf.reduce_mean(cross_entropyS) + tf.reduce_mean(cross_entropyE)

optimiser = tf.train.AdamOptimizer(0.001)
train_op = optimiser.minimize(loss)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    print("SESSION INITIALIZED")
    dataset_size = len(padded_data)
    padded_data = np.array(padded_data)
    np.random.shuffle(padded_data)
    #print("PADDED DATA SHAPE: ", padded_data.shape)
    padded_data_train = padded_data[0:(int) (0.95*padded_data.shape[0])]
    padded_data_validation = padded_data[(int) (0.95*padded_data.shape[0]):]
    
    print("Validating on",padded_data_validation.shape[0],"elements")

    losses = []
    for epoch in range(CONFIG.MAX_EPOCHS):
        print("Epoch # : ", epoch + 1)
        np.random.shuffle(padded_data)
        for iteration in range(0, len(padded_data_train) - CONFIG.BATCH_SIZE, CONFIG.BATCH_SIZE):
            batch = padded_data_train[iteration:iteration + CONFIG.BATCH_SIZE]
            question_batch = np.array(list(map(lambda qas: (qas["question"]), batch))).reshape(CONFIG.BATCH_SIZE,max_length_question)
            context_batch = np.array(list(map(lambda qas: (qas["context"]), batch))).reshape(CONFIG.BATCH_SIZE,max_length_context)
            question_mask_batch = np.array(list(map(lambda qas: (qas["question_mask"]), batch))).reshape(CONFIG.BATCH_SIZE,max_length_question)
            context_mask_batch = np.array(list(map(lambda qas: (qas["context_mask"]), batch))).reshape(CONFIG.BATCH_SIZE,max_length_context)
            answer_start_batch = np.array(list(map(lambda qas: (qas["answer_start"]), batch))).reshape(CONFIG.BATCH_SIZE)
            answer_end_batch = np.array(list(map(lambda qas: (qas["answer_end"]), batch))).reshape(CONFIG.BATCH_SIZE)
            _ , loss_val = sess.run([train_op,loss],feed_dict = {
                question_batch_placeholder : question_batch,
                context_batch_placeholder : context_batch,
                labels_placeholderS : answer_start_batch,
                labels_placeholderE : answer_end_batch,
                q_mask_placeholder: question_mask_batch,
                c_mask_placeholder: context_mask_batch,
                embedding: index2embedding
            })
            loss_val_mean = np.mean(loss_val)
            print("loss_val : ",loss_val_mean)
        mean_epoch_loss = np.mean(np.array(losses))
        print("loss: ", mean_epoch_loss)
