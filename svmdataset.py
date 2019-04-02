import sys
import numpy as np
import tensorflow as tf
from functools import reduce
import os
from preprocessing import answer_span_to_indices

# custom imports
from dataset import Dataset
from config import CONFIG
from build_model import get_batch
from evaluation_metrics import get_f1_from_tokens, get_exact_match_from_tokens
# Suppress tensorflow verboseness
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

D = Dataset('data/glove.6B.300d.txt')
index2embedding = D.index2embedding
padded_data_squad1, (max_length_question, max_length_context) = D.load_questions('data/train.json')
print("Loaded data from squad one")

padded_data_squad1 = np.array(padded_data_squad1)
padded_data_answerable = padded_data_squad1[(int) (CONFIG.TRAIN_PERCENTAGE * padded_data_squad1.shape[0]):]
padded_data_squad2, (max_length_question_squad2, max_length_context_squad2) = D.load_questions('data/train-v2.0.json')
print("Max length from squad 2 q and c: ", max_length_question_squad2, max_length_context_squad2)
print("Loaded data from squad two")

'''

latest_checkpoint_path = './model/saved-7' 
print("restoring from "+latest_checkpoint_path)
saver = tf.train.import_meta_graph(latest_checkpoint_path+'.meta')


with tf.Session(config=config) as sess:
    saver.restore(sess, latest_checkpoint_pa th)
    graph = tf.get_default_graph()
    answer_start_batch_predict = graph.get_tensor_by_name("answer_start:0")
    answer_end_batch_predict = graph.get_tensor_by_name("answer_end:0")
    question_batch_placeholder = graph.get_tensor_by_name("question_batch_ph:0")
    context_batch_placeholder = graph.get_tensor_by_name("context_batch_ph:0")
    embedding = graph.get_tensor_by_name("embedding_ph:0")
    dropout_keep_rate = graph.get_tensor_by_name("dropout_keep_ph:0")
    '''