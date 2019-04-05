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
from network.build_model import get_batch
from evaluation_metrics import get_f1_from_tokens, get_exact_match_from_tokens
# Suppress tensorflow verboseness
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


print("Starting testing on dev file...")
D = Dataset(CONFIG.EMBEDDING_FILE)
index2embedding = D.index2embedding
padded_data, (max_length_question, max_length_context) = D.load_questions('data/dev.json')
print("Loaded data")

# split padded data per the start index of the question
split_data_pre = dict()
for qas in padded_data: 
    first_word = D.index2word[qas["question"][0]].lower()
    if first_word not in split_data_pre:
        split_data_pre[first_word] = []
    split_data_pre[first_word].append(qas)

# Extract data bigger than batch size
split_data = dict()
print("First word frequency:")
for key in split_data_pre.keys():
    if len(split_data_pre[key]) > CONFIG.BATCH_SIZE:
        print(key + ': ' + str(len(split_data_pre[key])))
        split_data[key] = split_data_pre[key]

path_string = './model/saved-7'
latest_checkpoint_path = path_string

print("restoring from "+latest_checkpoint_path)
saver = tf.train.import_meta_graph(latest_checkpoint_path+'.meta')

config = tf.ConfigProto()
if '--noGPU' in sys.argv[1:]:
    print("Not using the GPU...")
    config = tf.ConfigProto(device_count = {'GPU': 0})

with tf.Session(config=config) as sess:
    saver.restore(sess, latest_checkpoint_path)
    graph = tf.get_default_graph()
    answer_start_batch_predict = graph.get_tensor_by_name("answer_start:0")
    answer_end_batch_predict = graph.get_tensor_by_name("answer_end:0")
    question_batch_placeholder = graph.get_tensor_by_name("question_batch_ph:0")
    context_batch_placeholder = graph.get_tensor_by_name("context_batch_ph:0")
    embedding = graph.get_tensor_by_name("embedding_ph:0")
    dropout_keep_rate = graph.get_tensor_by_name("dropout_keep_ph:0")
    alphas = graph.get_tensor_by_name("alphas:0")
    betas = graph.get_tensor_by_name("betas:0")


    print("SESSION INITIALIZED")
    f1_results = dict()
    for key in split_data.keys():
        print("testing questions with word " + key)
        padded_data = split_data[key]
        f1score = []
        for iteration in range(0, len(padded_data) - CONFIG.BATCH_SIZE, CONFIG.BATCH_SIZE):
            # running on an example batch to debug encoder
            batch = padded_data[iteration:(iteration + CONFIG.BATCH_SIZE)]
            question_batch, context_batch, answer_start_batch_actual, answer_end_batch_actual = get_batch(batch, CONFIG.BATCH_SIZE, max_length_question, max_length_context)
            answer = answer_span_to_indices(answer_start_batch_actual[0], answer_end_batch_actual[0], context_batch[0])

            estimated_start_index, estimated_end_index, s_logits, e_logits = sess.run([answer_start_batch_predict, answer_end_batch_predict, alphas, betas], feed_dict={
                question_batch_placeholder: question_batch,
                context_batch_placeholder: context_batch,
                embedding: index2embedding,
                dropout_keep_rate: 1
            })

            all_answers = np.array(list(map(lambda qas: (qas["all_answers"]), batch))).reshape(CONFIG.BATCH_SIZE)

            f1 = 0
            # Calculate f1 and em scores across batch size
            for i in range(CONFIG.BATCH_SIZE):
                f1_score_answers = []
                for true_answer in all_answers[i]:
                    f1_score_answers.append(get_f1_from_tokens(
                        true_answer["answer_start"],
                        true_answer["answer_end"],
                        estimated_start_index[i],
                        estimated_end_index[i],
                        context_batch[i],
                        D)
                    )
                f1 += max(f1_score_answers)

            f1score_curr = f1/CONFIG.BATCH_SIZE

            print("Current f1 score: ", f1score_curr)
            f1score.append(f1score_curr)

            print("Tested (",iteration,"/",len(padded_data),")")
        
        if len(f1score) != 0:
            results = dict()
            results["average"] = np.mean(f1score)
            results["max"] = np.max(f1score)
            results["min"] = np.min(f1score)
            f1_results[key] = results
            print("F1 mean for \'"+key+"\': " + str(f1_results[key]))

    with open('./results/test_question_split.pkl', 'wb') as f:
        pickle.dump(f1_results, f, protocol = 3)