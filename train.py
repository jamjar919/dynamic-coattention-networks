# This file trains the neural network using the encoder and decoder.
import sys
import numpy as np
import tensorflow as tf
import sklearn as sk
from functools import reduce
from sklearn.metrics import precision_score, recall_score, f1_score
import os

# custom imports
from encoder import encoder
from decoder import decoder
from dataset import Dataset

tensorboard_filepath = '.'

D = Dataset('data/train.json', 'data/glove.840B.300d.txt')
padded_data, index2embedding, max_length_question, max_length_context = D.load_data(sys.argv[1:])
print("Loaded data")

D = Dataset('data/dev.json', 'data/glove.840B.300d.txt')
padded_data_validation, index2embedding_validation, max_length_question_validation, max_length_context_validation = D.load_data(sys.argv[1:])
print("loaded validation data")

# Train now
batch_size = 128
embedding_dimension = 300
MAX_EPOCHS = 8
tf.reset_default_graph()

embedding = tf.placeholder(shape = [len(index2embedding), embedding_dimension], dtype=tf.float32, name='embedding')
question_batch_placeholder = tf.placeholder(dtype=tf.int32, shape = [batch_size, max_length_question], name='question_batch')
context_batch_placeholder = tf.placeholder(dtype=tf.int32, shape = [batch_size, max_length_context], name='context_batch')

# Create encoder
U = encoder(question_batch_placeholder,context_batch_placeholder,embedding)

# Word index placeholders
answer_start = tf.placeholder(dtype=tf.int32,shape=[None], name='answer_start_true')
answer_end = tf.placeholder(dtype=tf.int32,shape=[None], name='answer_end_true')

# Create decoder 
s, e, s_logits, e_logits = decoder(U)

s = tf.identity(s, name='answer_start')
e = tf.identity(e, name='answer_end')

l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_start,logits = s_logits)
l2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_end,logits = e_logits)

loss = l1 + l2
train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

hist_losses = []

saver = tf.train.Saver() 

init = tf.global_variables_initializer()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('./log_tensorboard', sess.graph)
    # Summaries need to be displayed
    # Whenever you need to record the loss, feed the mean loss to this placeholder
    tf_loss_ph = tf.placeholder(tf.float32,shape=None,name='loss_summary')
    # Create a scalar summary object for the loss so it can be displayed
    tf_loss_summary = tf.summary.scalar('loss', tf_loss_ph)

    sess.run(init)
    print("SESSION INITIALIZED")
    dataset_size = len(padded_data)
    padded_data = np.array(padded_data)
    for epoch in range(MAX_EPOCHS):
        print("Epoch # : ", epoch)
        # Shuffle the data between epochs
        np.random.shuffle(padded_data)
        for iteration in range(0, len(padded_data) - batch_size, batch_size):
            batch = padded_data[iteration:iteration + batch_size]
            question_batch = np.array(list(map(lambda qas: (qas["question"]), batch))).reshape(batch_size,max_length_question)
            context_batch = np.array(list(map(lambda qas: (qas["context"]), batch))).reshape(batch_size,max_length_context)
            answer_start_batch = np.array(list(map(lambda qas: (qas["answer_start"]), batch))).reshape(batch_size)
            answer_end_batch = np.array(list(map(lambda qas: (qas["answer_end"]), batch))).reshape(batch_size)
            _ , loss_val = sess.run([train_op,loss],feed_dict = {
                question_batch_placeholder : question_batch,
                context_batch_placeholder : context_batch,
                answer_start : answer_start_batch,
                answer_end : answer_end_batch,
                embedding: index2embedding
            })
        print("loss: ",np.mean(loss_val))
        summary_str = sess.run(tf_loss_summary, feed_dict={tf_loss_ph: np.mean(loss_val)})
        summary_writer.add_summary(summary_str,epoch)
        summary_writer.flush()

        # validation starting
        batch_size_validation = 128
        for counter in range(0, 256, batch_size_validation):
            # running on an example batch to debug encoder
            batch = padded_data_validation[counter:(counter + batch_size_validation)]
            question_batch_validation = np.array(list(map(lambda qas: (qas["question"]), batch))).reshape(batch_size_validation,
                                                                                               max_length_question_validation)
            context_batch_validation = np.array(list(map(lambda qas: (qas["context"]), batch))) \
                .reshape(batch_size_validation, max_length_context_validation)
            answer_start_batch_actual = np.array(list(map(lambda qas: (qas["answer_start"]), batch))) \
                .reshape(batch_size_validation)
            answer_end_batch_actual = np.array(list(map(lambda qas: (qas["answer_end"]), batch))).reshape(
                batch_size_validation)

            print("Running validation number = ", counter)
            s, e = sess.run([s_logits, e_logits], feed_dict={
                question_batch_placeholder: question_batch_validation,
                context_batch_placeholder: context_batch_validation,
                embedding: index2embedding_validation
            })

            estimated_start_index = np.argmax(s, axis = 1)
            estimated_end_index =  np.argmax(e, axis = 1)
            predictions = np.concatenate([estimated_start_index, estimated_end_index])
            actual = np.concatenate([answer_start_batch_actual, answer_end_batch_actual])
            #print("predictions", predictions)
            #print("actual", actual)
            print(
                "Precision", sk.metrics.precision_score(
                    predictions,
                    actual, average='micro')
            )
            print(
                "Recall", sk.metrics.recall_score(
                    predictions,
                    actual, average='micro')
            )
            print(
                "f1_score", sk.metrics.f1_score(
                    predictions,
                    actual, average='micro')
            )

        saver.save(sess, './model/saved', global_step=epoch)
    summary_writer.close()


