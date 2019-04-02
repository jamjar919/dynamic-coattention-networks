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

D = Dataset(CONFIG.EMBEDDING_FILE)
index2embedding = D.index2embedding
padded_data_squad1, (max_length_question, max_length_context) = D.load_questions('data/train.json')
padded_data_validation = padded_data_squad1[(int) (CONFIG.TRAIN_PERCENTAGE*len(padded_data_squad1)):]
untrained_contexts = [x["context"] for x in padded_data_validation]
print("Loaded data from squad one")

padded_data_squad2, (max_length_question_squad2, max_length_context_squad2) = D.load_questions('data/train-v2.0.json')
print("padded_data_squad2.len = ",len(padded_data_squad2))
print("Max length from squad 2 q and c: ", max_length_question_squad2, max_length_context_squad2)
print("Loaded data from squad two")

padded_data_untrained = [x for x in padded_data_squad2 if x["context"] in untrained_contexts]
unanswerable_data = [x for x in padded_data_untrained if x["answer_start"]==-1]
answerable_data = [x for x in padded_data_untrained if x["answer_start"]>=0]
print("Number of unanswerable questions: ",len(unanswerable_data))
print("Number of answerable questions: ", len(answerable_data))

padded_data = np.array(padded_data_untrained)
# # extract unanswerable questions
# padded_data_squad2 = np.array([x for x in padded_data_squad2 if x['answer_start'] == -1])
# print("Unanswerable questions: ",padded_data_squad2.shape[0])
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