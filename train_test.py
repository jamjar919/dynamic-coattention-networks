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

D = Dataset(CONFIG.EMBEDDING_FILE)
index2embedding = D.index2embedding
padded_data, (max_length_question, max_length_context) = D.load_questions('data/dev-v2.0.json')
print("Loaded data")

tf.reset_default_graph()
embedding = tf.placeholder(shape = [len(index2embedding), CONFIG.EMBEDDING_DIMENSION], dtype=tf.float32, name='embedding_ph')
train_op, loss, classifier_out  = build_classifier(embedding)

results_path = './resultsclassifier'
model_path = './modelclassifier'

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
        #np.random.shuffle(padded_data_train)
        for iteration in range(0, 1): #len(padded_data_train) - CONFIG.BATCH_SIZE, CONFIG.BATCH_SIZE):
            batch = padded_data_train[iteration:iteration + CONFIG.BATCH_SIZE]
            question_batch, context_batch, answer_batch = get_batch(batch, CONFIG.BATCH_SIZE, max_length_question, max_length_context)

            _ , loss_val = sess.run([train_op, loss],feed_dict = get_feed_dict(question_batch, context_batch, answer_batch, CONFIG.DROPOUT_KEEP_PROB, index2embedding))
            loss_value_mean = np.mean(loss_val)
            if(iteration % ((CONFIG.BATCH_SIZE)-1) == 0):
                print("Loss in epoch: ", loss_value_mean, "(",iteration,"/",len(padded_data_train),")")
            if(iteration % ((3*CONFIG.BATCH_SIZE)-1) == 0):
                print("Running validation...")
                batch = padded_data_validation[0:0 + CONFIG.BATCH_SIZE]
                question_batch_validation, context_batch_validation, has_answer_valid = get_batch(batch, CONFIG.BATCH_SIZE, max_length_question, max_length_context)
                answer_predicted = sess.run([classifier_out],
                    get_feed_dict(question_batch_validation,context_batch_validation,has_answer_valid, 1.0,  index2embedding))
                #answer_predicted = np.reshape(answer_predicted, shape = (answer_predicted.shape[0], answer_predicted.shape[1]))
                #print(answer_predicted)
                print(answer_predicted)
                print("Resuming training...")
