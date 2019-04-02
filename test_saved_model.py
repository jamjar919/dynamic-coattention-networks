# This file runs the actual question answering program using our trained network
import sys
import numpy as np
import tensorflow as tf
from functools import reduce
import os
from preprocessing import answer_span_to_indices

# custom imports
from dataset import Dataset
from network.config import CONFIG
from network.build_model import get_batch
from evaluation_metrics import get_f1_from_tokens, get_exact_match_from_tokens
# Suppress tensorflow verboseness
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


print("Starting testing on dev file...")
D = Dataset(CONFIG.EMBEDDING_FILE)
index2embedding = D.index2embedding
padded_data, (max_length_question, max_length_context) = D.load_questions('data/dev.json')
print("Loaded data")

latest_checkpoint_path = './model/saved-22' 
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
    # loss  = graph.get_tensor_by_name("loss_to_optimize:0")

    f1score = []
    emscore = []

    print("SESSION INITIALIZED")
    for iteration in range(0, len(padded_data) - CONFIG.BATCH_SIZE, CONFIG.BATCH_SIZE):
        # running on an example batch to debug encoder
        batch = padded_data[iteration:(iteration + CONFIG.BATCH_SIZE)]
        question_batch, context_batch, answer_start_batch_actual, answer_end_batch_actual = get_batch(batch, CONFIG.BATCH_SIZE, max_length_question, max_length_context)
        print("First context: ", D.index_to_text(context_batch[0]))
        print("First question: ", D.index_to_text(question_batch[0]))
        answer = answer_span_to_indices(answer_start_batch_actual[0], answer_end_batch_actual[0], context_batch[0])
        print("First answer label: ", D.index_to_text(answer))

        estimated_start_index, estimated_end_index = sess.run([answer_start_batch_predict, answer_end_batch_predict], feed_dict={
            question_batch_placeholder: question_batch,
            context_batch_placeholder: context_batch,
            embedding: index2embedding,
            dropout_keep_rate: 1
        })
        est_answer = answer_span_to_indices(estimated_start_index[0], estimated_end_index[0], context_batch[0])
        print("Predicted answer: ", D.index_to_text(est_answer))

        all_answers = np.array(list(map(lambda qas: (qas["all_answers"]), batch))).reshape(CONFIG.BATCH_SIZE)

        f1 = 0
        em = 0
        # Calculate f1 and em scores across batch size 
        for i in range(CONFIG.BATCH_SIZE):
            # maximise f1 score across answers
            f1_score_answers = []
            em_score_answers = []
            for true_answer in all_answers[i]:
                f1_score_answers.append(get_f1_from_tokens(
                    true_answer["answer_start"],
                    true_answer["answer_end"],
                    estimated_start_index[i],
                    estimated_end_index[i],
                    context_batch[i],
                    D)
                )

                em_score_answers.append(get_exact_match_from_tokens(
                    true_answer["answer_start"],
                    true_answer["answer_end"],
                    estimated_start_index[i],
                    estimated_end_index[i],
                    context_batch[i],
                    D)
                )

            f1 += max(f1_score_answers)
            em += max(em_score_answers)
    
        f1score_curr = f1/CONFIG.BATCH_SIZE
        emscore_curr = em/CONFIG.BATCH_SIZE
        print("Current f1 score: ", f1score_curr)
        print("Current em score: ", emscore_curr)
        f1score.append(f1score_curr)
        emscore.append(emscore_curr)

        #if(iteration % ((CONFIG.BATCH_SIZE)-1) == 0):
        print("Tested (",iteration,"/",len(padded_data),")")

    print("F1 mean: ", np.mean(f1score))
    print("EM mean: ", np.mean(emscore))