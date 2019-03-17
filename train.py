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

D_train = Dataset('data/train.json', 'data/glove.840B.300d.txt')
padded_data, index2embedding, max_length_question, max_length_context = D_train.load_data(sys.argv[1:])
print("Loaded data")

# Train now
batch_size = 64
embedding_dimension = 300
MAX_EPOCHS = 40
tf.reset_default_graph()

embedding = tf.placeholder(shape = [len(index2embedding), embedding_dimension], dtype=tf.float32, name='embedding')
question_batch_placeholder = tf.placeholder(dtype=tf.int32, shape = [batch_size, max_length_question], name='question_batch')
context_batch_placeholder = tf.placeholder(dtype=tf.int32, shape = [batch_size, max_length_context], name='context_batch')

# Create encoder. (Encoder will also return the sequence length of the context (i.e. how much of each batch element is unpadded))
U, seq_length = encoder(question_batch_placeholder,context_batch_placeholder,embedding)
#context_ph_length = tf.Print(context_ph_length, [context_ph_length.shape], "Context lengths: ")
# Word index placeholders
answer_start = tf.placeholder(dtype=tf.int32,shape=[None], name='answer_start_true')
answer_end = tf.placeholder(dtype=tf.int32,shape=[None], name='answer_end_true')

# Create decoder 
s, e, s_logits, e_logits = decoder(U, seq_length, max_length_context) # Pass also the seq_length from encoder and max_length.

s = tf.identity(s, name='answer_start')
e = tf.identity(e, name='answer_end')

l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_start,logits = s_logits)
l2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_end,logits = e_logits)

loss = l1 + l2
train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

saver = tf.train.Saver() 
init = tf.global_variables_initializer()

with tf.Session() as sess:
    loss_writer = tf.summary.FileWriter('./log_tensorboard/plot_loss', sess.graph)
    val_writer = tf.summary.FileWriter('./log_tensorboard/plot_val', sess.graph)
    # Summaries need to be displayed
    # Whenever you need to record the loss, feed the mean loss to this placeholder
    tf_loss_ph = tf.placeholder(tf.float32, shape=None,name='Loss_summary')
    tf_validation_ph = tf.placeholder(tf.float32, shape = None, name = 'f1_score')
    # Create a scalar summary object for the loss so it can be displayed on tensorboard
    tf_loss_summary = tf.summary.scalar('Loss_summary', tf_loss_ph)
    tf_validation_summary = tf.summary.scalar('f1_score', tf_validation_ph)
    sess.run(init)
    print("SESSION INITIALIZED")
    dataset_size = len(padded_data)
    padded_data = np.array(padded_data)
    np.random.shuffle(padded_data)
    #print("PADDED DATA SHAPE: ", padded_data.shape)
    padded_data_train = padded_data[0:(int) (0.95*padded_data.shape[0])]
    padded_data_validation = padded_data[(int) (0.95*padded_data.shape[0]):]
    
    for epoch in range(MAX_EPOCHS):
        print("Epoch # : ", epoch + 1)
        # Shuffle the data between epochs
        np.random.shuffle(padded_data)
        for iteration in range(0, len(padded_data_train) - batch_size, batch_size):
            batch = padded_data_train[iteration:iteration + batch_size]
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
        loss_writer.add_summary(summary_str,epoch)
        loss_writer.flush()
        
        f1score = []
        #precision = []
        #recall = []
        # validation starting
        for counter in range(0, len(padded_data_validation) - batch_size, batch_size):
            batch = padded_data_validation[counter:(counter + batch_size)]
            question_batch_validation = np.array(list(map(lambda qas: (qas["question"]), batch))).reshape(batch_size,
                                                                                               max_length_question)
            context_batch_validation = np.array(list(map(lambda qas: (qas["context"]), batch))) \
                .reshape(batch_size, max_length_context)
            answer_start_batch_actual = np.array(list(map(lambda qas: (qas["answer_start"]), batch))) \
                .reshape(batch_size)
            answer_end_batch_actual = np.array(list(map(lambda qas: (qas["answer_end"]), batch))).reshape(
                batch_size)

            s, e = sess.run([s_logits, e_logits], feed_dict={
                question_batch_placeholder: question_batch_validation,
                context_batch_placeholder: context_batch_validation,
                embedding: index2embedding
            })

            estimated_start_index = np.argmax(s, axis = 1)
            estimated_end_index =  np.argmax(e, axis = 1)
            predictions = np.concatenate([estimated_start_index, estimated_end_index])
            actual = np.concatenate([answer_start_batch_actual, answer_end_batch_actual])

            #precision.append(sk.metrics.precision_score(predictions, actual, average='micro'))
            #recall.append(sk.metrics.recall_score(predictions,actual, average='micro'))
            f1score.append(sk.metrics.f1_score(predictions, actual, average = 'micro'))
        f1_mean = np.mean(np.array(f1score))
        print("Validation f1 score %: ", f1_mean * 100)
        summary_str = sess.run(tf_validation_summary, feed_dict={tf_validation_ph: f1_mean})
        val_writer.add_summary(summary_str, epoch)
        val_writer.flush()
        saver.save(sess, './model/saved', global_step=epoch)
    loss_writer.close()