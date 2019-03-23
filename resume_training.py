# This file runs the actual question answering program using our trained network
import sys
import numpy as np
import tensorflow as tf
import sklearn as sk
from config import CONFIG

# custom imports
from dataset import Dataset
from evaluation_metrics import *

print("Resumeing training")
D_train = Dataset(CONFIG.QUESTION_FILE, CONFIG.EMBEDDING_FILE)
padded_data, index2embedding, max_length_question, max_length_context = D_train.load_data(sys.argv[1:])
print("Loaded data")

saver = tf.train.Saver()
tf.reset_default_graph()
imported_graph = tf.train.import_meta_graph('./model/saved-6.meta')

init = tf.global_variables_initializer()
batch_size = CONFIG.BATCH_SIZE

with tf.Session() as sess:

    imported_graph.restore(sess, './model/saved-17')
    graph = tf.get_default_graph()

    question_batch_placeholder = graph.get_tensor_by_name("question_batch:0")
    context_batch_placeholder = graph.get_tensor_by_name("context_batch:0")
    answer_start = graph.get_tensor_by_name("answer_start_true:0")
    answer_end = graph.get_tensor_by_name("answer_end_true:0")
    loss = graph.get_tensor_by_name("loss_to_optimize:0")
    train_op = graph.get_tensor_by_name("train_op:0")
    embedding = graph.get_tensor_by_name("embedding:0")

    loss_writer = tf.summary.FileWriter('./log_tensorboard/plot_loss', sess.graph)
    val_writer = tf.summary.FileWriter('./log_tensorboard/plot_val', sess.graph)
    # Summaries need to be displayed
    # Whenever you need to record the loss, feed the mean loss to this placeholder
    tf_loss_ph = tf.placeholder(tf.float32, shape=None, name='Loss_summary')
    tf_validation_ph = tf.placeholder(tf.float32, shape=None, name='f1_score')
    # Create a scalar summary object for the loss so it can be displayed on tensorboard
    tf_loss_summary = tf.summary.scalar('Loss_summary', tf_loss_ph)
    tf_validation_summary = tf.summary.scalar('f1_score', tf_validation_ph)
    sess.run(init)
    print("SESSION INITIALIZED")
    dataset_size = len(padded_data)
    padded_data = np.array(padded_data)
    np.random.shuffle(padded_data)
    # print("PADDED DATA SHAPE: ", padded_data.shape)
    padded_data_train = padded_data[0:(int)(0.95 * padded_data.shape[0])]
    padded_data_validation = padded_data[(int)(0.95 * padded_data.shape[0]):]

    print("Validating on", padded_data_validation.shape[0], "elements")

    losses = []
    for epoch in range(CONFIG.MAX_EPOCHS):
        print("Epoch # : ", epoch + 1)
        # Shuffle the data between epochs
        np.random.shuffle(padded_data)
        for iteration in range(0, len(padded_data_train) - CONFIG.BATCH_SIZE, CONFIG.BATCH_SIZE):
            batch = padded_data_train[iteration:iteration + CONFIG.BATCH_SIZE]
            question_batch = np.array(list(map(lambda qas: (qas["question"]), batch))).reshape(CONFIG.BATCH_SIZE,
                                                                                               max_length_question)
            context_batch = np.array(list(map(lambda qas: (qas["context"]), batch))).reshape(CONFIG.BATCH_SIZE,
                                                                                             max_length_context)
            answer_start_batch = np.array(list(map(lambda qas: (qas["answer_start"]), batch))).reshape(
                CONFIG.BATCH_SIZE)
            answer_end_batch = np.array(list(map(lambda qas: (qas["answer_end"]), batch))).reshape(CONFIG.BATCH_SIZE)
            _, loss_val, alpha_logits, beta_logits = sess.run([train_op, loss], feed_dict={
                question_batch_placeholder: question_batch,
                context_batch_placeholder: context_batch,
                answer_start: answer_start_batch,
                answer_end: answer_end_batch,
                embedding: index2embedding
            })
            loss_val_mean = np.mean(loss_val)
            print("loss_val : ", loss_val_mean)
            # for alpha_logit in alpha_logits :
            #     print(alpha_logit)
            losses.append(loss_val_mean.item())
        mean_epoch_loss = np.mean(np.array(losses))
        print("loss: ", mean_epoch_loss)
        summary_str = sess.run(tf_loss_summary, feed_dict={tf_loss_ph: mean_epoch_loss})
        loss_writer.add_summary(summary_str, epoch)
        loss_writer.flush()

        f1score = []
        validation_losses = []

        # validation starting
        for counter in range(0, len(padded_data_validation) - CONFIG.BATCH_SIZE, CONFIG.BATCH_SIZE):
            batch = padded_data_validation[counter:(counter + CONFIG.BATCH_SIZE)]
            question_batch_validation = np.array(list(map(lambda qas: (qas["question"]), batch))).reshape(
                CONFIG.BATCH_SIZE,
                max_length_question)
            context_batch_validation = np.array(list(map(lambda qas: (qas["context"]), batch))) \
                .reshape(CONFIG.BATCH_SIZE, max_length_context)
            answer_start_batch_actual = np.array(list(map(lambda qas: (qas["answer_start"]), batch))) \
                .reshape(CONFIG.BATCH_SIZE)
            answer_end_batch_actual = np.array(list(map(lambda qas: (qas["answer_end"]), batch))).reshape(
                CONFIG.BATCH_SIZE)

            estimated_start_index, estimated_end_index, loss_validation = sess.run([s, e, loss],
                                                                                   feed_dict={
                                                                                       question_batch_placeholder: question_batch_validation,
                                                                                       context_batch_placeholder: context_batch_validation,
                                                                                       answer_start: answer_start_batch_actual,
                                                                                       answer_end: answer_end_batch_actual,
                                                                                       embedding: index2embedding
                                                                                   })

            # print("pred:", loss_validation, estimated_start_index,"->" , estimated_end_index)
            # print("real:", loss_validation, answer_start_batch_actual,"->", answer_end_batch_actual)

            predictions = np.concatenate([estimated_start_index, estimated_end_index])
            actual = np.concatenate([answer_start_batch_actual, answer_end_batch_actual])

            validation_losses.append(loss_validation)
            f1score.append(sk.metrics.f1_score(predictions, actual, average='micro'))

            f1 = 0
            for i in range(len(estimated_end_index)):
                f1 += get_f1_from_tokens(answer_start_batch_actual[i], answer_end_batch_actual[i],
                                         estimated_start_index[i], estimated_end_index[i],
                                         context_batch_validation[i])
            print("f1 score: ", f1 / len(estimated_end_index))

        # print(f1score)
        f1_mean = np.mean(f1score)
        validation_loss = np.mean(validation_losses)
        print("Validation loss: ", validation_loss)
        print("Validation f1 score %: ", f1_mean * 100)
        summary_str = sess.run(tf_validation_summary, feed_dict={tf_validation_ph: f1_mean})
        val_writer.add_summary(summary_str, epoch)
        val_writer.flush()
        saver.save(sess, './model/saved', global_step=epoch)
