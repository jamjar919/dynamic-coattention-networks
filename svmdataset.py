import sys
import numpy as np
import tensorflow as tf
import pickle
from functools import reduce
import os
from preprocessing import answer_span_to_indices
from scipy.special import softmax
# custom imports
from dataset import Dataset
from network.config import CONFIG
from network.build_model import get_batch
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

latest_checkpoint_path = './model/saved-7' 
print("restoring from "+latest_checkpoint_path)
saver = tf.train.import_meta_graph(latest_checkpoint_path+'.meta')

config = tf.ConfigProto()
if '--noGPU' in sys.argv[1:]:
    print("Not using the GPU...")
    config = tf.ConfigProto(device_count = {'GPU': 0})

with tf.Session(config=config) as sess:
    saver.restore(sess, latest_checkpoint_path)
    graph = tf.get_default_graph()
    alphas_predict = graph.get_tensor_by_name("alphas:0")
    betas_predict = graph.get_tensor_by_name("betas:0")
    question_batch_placeholder = graph.get_tensor_by_name("question_batch_ph:0")
    context_batch_placeholder = graph.get_tensor_by_name("context_batch_ph:0")
    embedding = graph.get_tensor_by_name("embedding_ph:0")
    dropout_keep_rate = graph.get_tensor_by_name("dropout_keep_ph:0")

    alpha_raw, beta_raw, labels = [], [], []

    for iteration in range(0, len(padded_data) - CONFIG.BATCH_SIZE, CONFIG.BATCH_SIZE):
            # running on an example batch to debug encoder
            batch = padded_data[iteration : (iteration + CONFIG.BATCH_SIZE)]
            question_batch, context_batch, answer_start_batch_actual, answer_end_batch_actual = get_batch(batch, CONFIG.BATCH_SIZE, max_length_question, max_length_context)

            alphas, betas = sess.run([alphas_predict, betas_predict], feed_dict={
                question_batch_placeholder: question_batch,
                context_batch_placeholder: context_batch,
                embedding: index2embedding,
                dropout_keep_rate: 1.0
            })
            #print(alphas.shape)
            for i in range(answer_start_batch_actual.shape[0]):
                alpha_raw.append(softmax(alphas[-1][i][:]))
                beta_raw.append(softmax(betas[-1][i][:]))
                
                if answer_start_batch_actual[i] == -1:
                    labels.append(-1)
                else:
                    labels.append(1)
  
            #print(len(alpha_raw))

    with open('./SVMdata/alphas_raw.pkl', 'wb') as f:
        pickle.dump(alpha_raw, f, protocol=3)
    with open('./SVMdata/betas_raw.pkl', 'wb') as f:
        pickle.dump(beta_raw, f, protocol=3)
    with open('./SVMdata/labels.pkl', 'wb') as f:
        pickle.dump(labels, f, protocol = 3)