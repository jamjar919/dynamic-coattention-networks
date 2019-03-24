# This file trains the neural network using the encoder and decoder.
import sys
import numpy as np
import tensorflow as tf
import pickle
from functools import reduce
from sklearn.metrics import precision_score, recall_score, f1_score
import os
# custom imports
from encoder import encoder
from decoder import decoder
from dataset import Dataset
from config import CONFIG
from evaluation_metrics import *
from build_model import build_model, get_feed_dict, get_batch

tensorboard_filepath = '.'

D_train = Dataset(CONFIG.QUESTION_FILE, CONFIG.EMBEDDING_FILE)
padded_data, index2embedding, max_length_question, max_length_context = D_train.load_data(sys.argv[1:])
print("Loaded data")

tf.reset_default_graph()
embedding = tf.placeholder(shape = [len(index2embedding), CONFIG.EMBEDDING_DIMENSION], dtype=tf.float32, name='embedding_ph')
train_op, loss, s, e  = build_model(embedding)

# Train now
saver = tf.train.Saver() 
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
    
    print("LEN PADDED DATA TRAIN: ", len(padded_data_train))
    loss_means = []
    val_loss_means = []
    val_f1_means = []
    for epoch in range(CONFIG.MAX_EPOCHS):
        print("Epoch # : ", epoch + 1)
        losses = []
        # Shuffle the data between epochs
        np.random.shuffle(padded_data)
        for iteration in range(0, len(padded_data_train) - CONFIG.BATCH_SIZE, CONFIG.BATCH_SIZE):
            batch = padded_data_train[iteration:iteration + CONFIG.BATCH_SIZE]
            question_batch, context_batch, answer_start_batch, answer_end_batch = get_batch(batch, CONFIG.BATCH_SIZE, max_length_question, max_length_context)

            _ , loss_val = sess.run([train_op, loss],feed_dict = get_feed_dict(question_batch,context_batch,answer_start_batch,answer_end_batch, CONFIG.DROPOUT_KEEP_PROB, index2embedding))
            loss_val_mean = np.mean(loss_val)
            if(iteration % ((CONFIG.BATCH_SIZE)-1) == 0):
                print("Loss in epoch: ", loss_val_mean, "(",iteration,"/",len(padded_data_train),")")

            losses.append(loss_val_mean.item())
        mean_epoch_loss = np.mean(np.array(losses))
        loss_means.append(mean_epoch_loss)
        print("Mean epoch loss: ", mean_epoch_loss)
        f1score = []
        validation_losses = []

        #validation starting
        for counter in range(0, len(padded_data_validation) - CONFIG.BATCH_SIZE, CONFIG.BATCH_SIZE):
            batch = padded_data_validation[counter:(counter + CONFIG.BATCH_SIZE)]
            question_batch_validation, context_batch_validation, answer_start_batch_actual, answer_end_batch_actual = get_batch(batch, CONFIG.BATCH_SIZE, max_length_question, max_length_context)

            estimated_start_index, estimated_end_index, loss_validation = sess.run([s, e, loss],
            get_feed_dict(question_batch_validation,context_batch_validation,answer_end_batch_actual,answer_end_batch_actual, 1.0, index2embedding)
            )

            validation_losses.append(np.mean(loss_validation))
            f1 = 0            
            for i in range(len(estimated_end_index)):
                #print("start actual, end actual, start pred, end pred: ", answer_start_batch_actual[i], answer_end_batch_actual[i], estimated_start_index[i], estimated_end_index[i])
                f1 += get_f1_from_tokens(answer_start_batch_actual[i], answer_end_batch_actual[i],
                                   estimated_start_index[i], estimated_end_index[i],
                                   context_batch_validation[i], D_train)
            f1score.append(f1/len(estimated_end_index))
            #print("f1 score: ", f1/len(estimated_end_index))
  
        print("F1 mean on validation: ", np.mean(f1score))
        print("Mean validation loss on epoch: ", np.mean(validation_losses))
        val_loss_means.append(np.mean(validation_losses))
        val_f1_means.append(np.mean(f1score))

        with open('./results/validation_loss_means.pkl', 'wb') as f:
            pickle.dump(val_loss_means, f, protocol=3)
        with open('./results/validation_f1_means.pkl', 'wb') as f:
            pickle.dump(val_f1_means, f, protocol=3)
        with open('./results/training_loss_means.pkl', 'wb') as f:
            pickle.dump(loss_means, f, protocol=3)

        saver.save(sess, './model/saved', global_step=epoch)

