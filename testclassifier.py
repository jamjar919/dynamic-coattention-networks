# This file runs the actual question answering program using our trained network
import sys
import numpy as np
import tensorflow as tf
from functools import reduce
import os
import pickle
from preprocessing import answer_span_to_indices
# custom imports
from dataset import Dataset
from network.config import CONFIG
from network.classifier import get_batch, get_feed_dict
from network.build_model import get_batch as get_batch_span
from score import Score
# Suppress tensorflow verboseness
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


print("Starting testing on dev file...")
D = Dataset(CONFIG.EMBEDDING_FILE)
index2embedding = D.index2embedding
padded_data, (max_length_question, max_length_context) = D.load_questions('data/dev-v2.0.json')
print("Loaded data")

model_path = './modelclassifier'
results_path = './resultsclassifier'

path_string = model_path + '/saved-' + str(3)
latest_checkpoint_path = path_string

print("restoring from " + latest_checkpoint_path)
saver = tf.train.import_meta_graph(latest_checkpoint_path+'.meta')

config = tf.ConfigProto()
if '--noGPU' in sys.argv[1:]:
    print("Not using the GPU...")
    config = tf.ConfigProto(device_count = {'GPU': 0})

with tf.Session(config=config) as sess:
    saver.restore(sess, latest_checkpoint_path)
    graph = tf.get_default_graph()
    answer_predict = graph.get_tensor_by_name("classifier_output:0")
    question_batch_placeholder = graph.get_tensor_by_name("question_batch_ph:0")
    context_batch_placeholder = graph.get_tensor_by_name("context_batch_ph:0")
    embedding = graph.get_tensor_by_name("embedding_ph:0")
    dropout_keep_rate = graph.get_tensor_by_name("dropout_keep_ph:0")
    loss  = graph.get_tensor_by_name("loss_v2_classifier:0")

    score_50 = Score()
    score_67 = Score()
    score_80 = Score()
    score_30 = Score()
    score_5 = Score()
    score_95 = Score()
    losses = []
    predicted_labels = []
    actual_labels = []
    print("SESSION INITIALIZED")
    for iteration in range(0, len(padded_data) - CONFIG.BATCH_SIZE, CONFIG.BATCH_SIZE):
        # running on an example batch to debug encoder
        batch = padded_data[iteration:(iteration + CONFIG.BATCH_SIZE)]
        question_batch, context_batch, answer_actual = get_batch(batch, CONFIG.BATCH_SIZE, max_length_question, max_length_context)
        #_ , _ , answer_start_batch_actual, answer_end_batch_actual = get_batch_span(batch, CONFIG.BATCH_SIZE, max_length_question, max_length_context)

        yhat, loss_value = sess.run([answer_predict, loss], 
            get_feed_dict(question_batch,context_batch,answer_actual, 1.0,  index2embedding))

        yhat = np.reshape(yhat, (yhat.shape[0], yhat.shape[1]))
        #print(yhat.shape)
        #print("predicted values: ", yhat)
        print("current loss: ", np.mean(loss_value), "(",iteration,"/",len(padded_data),")")
        for i in range (yhat.shape[0]):
            predicted_labels.append(yhat[i])
            actual_labels.append(answer_actual[i])
        
        losses.append(np.mean(loss_value))
        predicted_labels_50 = np.where(yhat > 0.5, 1, 0)
        predicted_labels_67 = np.where(yhat > 0.67, 1, 0)
        predicted_labels_80 = np.where(yhat > 0.80, 1, 0)
        predicted_labels_30 = np.where(yhat > 0.3, 1, 0)
        predicted_labels_5 = np.where(yhat > 0.05, 1, 0)
        predicted_labels_95 = np.where(yhat > 0.95, 1, 0)
        score_50.update(predicted_labels_50, answer_actual)
        score_67.update(predicted_labels_67, answer_actual)
        score_80.update(predicted_labels_80, answer_actual)
        score_30.update(predicted_labels_30, answer_actual)
        score_95.update(predicted_labels_95, answer_actual)
        score_5.update(predicted_labels_5, answer_actual)

    score50_file = results_path + '/score50.pkl'
    score67_file = results_path + '/score67.pkl'
    score30_file = results_path + '/score30.pkl'
    score80_file = results_path + '/score80.pkl'
    score5_file = results_path + '/score5.pkl'
    score95_file = results_path + '/score95.pkl'
    print("Mean accuracy using 67 threshold: ", score_67.accuracy)
    print("Mean accuracy using 50 threshold: ", score_50.accuracy)
    print("Mean accuracy using 80 threshold: ", score_80.accuracy)
    print("Mean accuracy using 30 threshold: ", score_30.accuracy)
    print("Mean accuracy using 5 threshold: ", score_5.accuracy)
    print("Mean accuracy using 95 threshold: ", score_95.accuracy)
    print("Mean testing loss: ", np.mean(losses))
    append_write = 'wb'  # make a new file if not
    with open(score50_file, append_write) as f:
        pickle.dump(score_50, f, protocol=3)
    with open(score67_file, append_write) as f:
        pickle.dump(score_67, f, protocol=3)
    with open(score80_file, append_write) as f:
        pickle.dump(score_80, f, protocol=3)
    with open(score30_file, append_write) as f:
        pickle.dump(score_30, f, protocol=3)
    with open(results_path + '/predicted_outputs.pkl', append_write) as f:
        pickle.dump(predicted_labels, f, protocol=3)
    with open(results_path + '/actual_labels.pkl', append_write) as f:
        pickle.dump(actual_labels, f, protocol=3)
    with open(results_path + '/test_losses.pkl', append_write) as f:
        pickle.dump(losses, f, protocol=3) 
    with open(score5_file, append_write) as f:
        pickle.dump(score_5, f, protocol=3)
    with open(score95_file, append_write) as f:
        pickle.dump(score_95, f, protocol=3) 