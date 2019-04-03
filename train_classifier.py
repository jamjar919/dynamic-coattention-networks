# This file trains the neural network using the encoder and decoder.
import sys
import numpy as np
import tensorflow as tf
import pickle
from functools import reduce
import os
# custom imports
from network.config import CONFIG
from network.classifier import build_classifier, get_batch, get_feed_dict
from evaluation_metrics import get_f1_from_tokens, get_exact_match_from_tokens
from dataset import Dataset
from score import Score

tensorboard_filepath = '.'

D = Dataset('data/glove.6B.300d.txt') #(CONFIG.EMBEDDING_FILE)
index2embedding = D.index2embedding
padded_data, (max_length_question, max_length_context) = D.load_questions(CONFIG.QUESTION_FILE_V2)
print("Loaded data")

tf.reset_default_graph()
embedding = tf.placeholder(shape = [len(index2embedding), CONFIG.EMBEDDING_DIMENSION], dtype=tf.float32, name='embedding_ph')
train_op, loss, classifier_out  = build_classifier(embedding)

results_path = './resultsv2'
model_path = './modelv2'

config = tf.ConfigProto()
if '--noGPU' in sys.argv[1:]:
    print("Not using the GPU...")
    config = tf.ConfigProto(device_count = {'GPU': 0})
    
# Train now
saver = tf.train.Saver(max_to_keep = CONFIG.MAX_EPOCHS) 
init = tf.global_variables_initializer()


THRESHOLD = 0.5
with tf.Session(config=config) as sess:
    sess.run(init)
    print("SESSION INITIALIZED")
    dataset_size = len(padded_data)
    padded_data = np.array(padded_data)
    
    #print("PADDED DATA SHAPE: ", padded_data.shape)
    padded_data_train = padded_data[0:(int) (CONFIG.TRAIN_PERCENTAGE*padded_data.shape[0])]
    padded_data_validation = padded_data[(int) (CONFIG.TRAIN_PERCENTAGE*padded_data.shape[0]):]
    np.random.shuffle(padded_data_train)
    np.random.shuffle(padded_data_validation)

    print("LEN PADDED DATA TRAIN: ", len(padded_data_train))
    losses_epoch = []
    validation_epoch_loss = []
    val_scores = []
    score = Score()
    for epoch in range(CONFIG.MAX_EPOCHS):
        print("Epoch # : ", epoch + 1)
        losses_in_epoch = []
        # Shuffle the data between epochs
        np.random.shuffle(padded_data_train)
        for iteration in range(0, 2): #len(padded_data_train) - CONFIG.BATCH_SIZE, CONFIG.BATCH_SIZE):
            batch = padded_data_train[iteration:iteration + CONFIG.BATCH_SIZE]
            question_batch, context_batch, answer_batch = get_batch(batch, CONFIG.BATCH_SIZE, max_length_question, max_length_context)

            _ , loss_val = sess.run([train_op, loss],feed_dict = get_feed_dict(question_batch, context_batch, answer_batch, CONFIG.DROPOUT_KEEP_PROB, index2embedding))
            loss_value_mean = np.mean(loss_val)
            if(iteration % ((CONFIG.BATCH_SIZE)-1) == 0):
                print("Loss in epoch: ", loss_value_mean, "(",iteration,"/",len(padded_data_train),")")

            losses_in_epoch.append(loss_value_mean.item())
        losses_epoch.append(np.mean(np.array(losses_in_epoch)))
        print("Mean epoch loss: ", np.mean(np.array(losses_in_epoch)))

        validation_loss_in_epoch = []
        score.reset()
        #validation starting
        for counter in range(0, 2): #len(padded_data_validation) - CONFIG.BATCH_SIZE, CONFIG.BATCH_SIZE):
            batch = padded_data_validation[counter:(counter + CONFIG.BATCH_SIZE)]
            question_batch_validation, context_batch_validation, has_answer_valid = get_batch(batch, CONFIG.BATCH_SIZE, max_length_question, max_length_context)

            answer_predicted, loss_validation = sess.run([classifier_out, loss],
            get_feed_dict(question_batch_validation,context_batch_validation,has_answer_valid, 1.0,  index2embedding))
            validation_loss_in_epoch.append(np.mean(loss_validation))
            predicted_labels = np.where(answer_predicted > THRESHOLD, 1, 0)
            print("predicted labels shape": predicted_labels.shape)
            score.update(predicted_labels, has_answer_valid)

        print("Validation loss of epoch: ", np.mean(validation_loss_in_epoch))
        print("Validation accuracy score of epoch: ", score.accuracy)
        print("Validation precision score of epoch: ", score.precision)
        print("Validation F1 score of epoch: ", score.F1)
        validation_epoch_loss.append(np.mean(validation_loss_in_epoch))
        val_scores.append(score)

        with open (results_path+'/validation_scores_classifier.pkl', 'wb') as f :
            pickle.dump(val_scores, f, protocol = 3) 
        with open(results_path + '/validation_loss_means_classifier.pkl', 'wb') as f:
            pickle.dump(validation_epoch_loss, f, protocol=3)
        with open(results_path + '/training_loss_means_classifier.pkl', 'wb') as f:
            pickle.dump(losses_epoch, f, protocol = 3)


        saver.save(sess, model_path + '/saved', global_step=epoch)

