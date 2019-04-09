import __init__
import sys
import numpy as np
import tensorflow as tf
from functools import reduce
import os
import pickle
from preprocessing.preprocessing import answer_span_to_indices

# custom imports
from preprocessing.dataset import Dataset
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

config = tf.ConfigProto()
if '--noGPU' in sys.argv[1:]:
    print("Not using the GPU...")
    config = tf.ConfigProto(device_count = {'GPU': 0})

model_path = './model'
results_path = './results'

for i in range(0, 12):
    path_string = model_path + './saved-' + str(i)
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
        answer_start_true = graph.get_tensor_by_name("answer_start_true_ph:0")
        answer_end_true = graph.get_tensor_by_name("answer_end_true_ph:0")
        question_batch_placeholder = graph.get_tensor_by_name("question_batch_ph:0")
        context_batch_placeholder = graph.get_tensor_by_name("context_batch_ph:0")
        embedding = graph.get_tensor_by_name("embedding_ph:0")
        dropout_keep_rate = graph.get_tensor_by_name("dropout_keep_ph:0")
        alphas = graph.get_tensor_by_name("alphas:0")
        betas = graph.get_tensor_by_name("betas:0")
        loss  = graph.get_tensor_by_name("loss:0")

        f1score = []
        emscore = []
        losses_list = []

        print("SESSION INITIALIZED")
        for iteration in range(0, len(padded_data) - CONFIG.BATCH_SIZE, CONFIG.BATCH_SIZE):
            # running on an example batch to debug encoder
            batch = padded_data[iteration:(iteration + CONFIG.BATCH_SIZE)]
            question_batch, context_batch, answer_start_batch_actual, answer_end_batch_actual = get_batch(batch, CONFIG.BATCH_SIZE, max_length_question, max_length_context)

            losses, estimated_start_index, estimated_end_index, s_logits, e_logits = sess.run([loss, answer_start_batch_predict, answer_end_batch_predict, alphas, betas], feed_dict={
                question_batch_placeholder: question_batch,
                context_batch_placeholder: context_batch,
                answer_start_true : answer_start_batch_actual,
                answer_end_true : answer_end_batch_actual,
                embedding: index2embedding,
                dropout_keep_rate: 1
            })
            losses_list.append(np.mean(losses))

            all_answers = np.array(list(map(lambda qas: (qas["all_answers"]), batch))).reshape(CONFIG.BATCH_SIZE)

            f1 = 0
            em = 0
            alpha_loss_total = 0
            beta_loss_total = 0
            ignored_alpha_losses = 0
            ignored_beta_losses = 0
            # Calculate f1 and em scores across batch size
            for i in range(CONFIG.BATCH_SIZE):
                # print(np.sum(np.exp(s_logits)))
                # print(np.exp(s_logits))
                # print( np.log(np.exp(s_logits)/np.sum(np.exp(s_logits))))
                #print(s_logits.shape)
                #print(len(np.exp(s_logits[i])))
                alpha_loss = 0
                beta_loss = 0
                for j in range(4):
                    alpha_loss += -1. * np.log(np.exp(s_logits[j][i])/np.sum(np.exp(s_logits[j][i])))[all_answers[i][0]["answer_start"]]
                    beta_loss += -1. * np.log(np.exp(e_logits[j][i]) / np.sum(np.exp(e_logits[j][i])))[all_answers[i][0]["answer_end"]]

                #print(str(alpha_loss) + "THIS IS alpha" + str(i))
                #print(str(beta_loss) + "THIS IS " + str(i))
                if alpha_loss == float("inf") or alpha_loss == float("-inf"):
                    ignored_alpha_losses += 1
                else:
                    alpha_loss_total += alpha_loss

                if beta_loss == float("inf") or beta_loss == float("-inf"):
                    ignored_beta_losses += 1
                else:
                    beta_loss_total += beta_loss
                print("*****")
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

            alpha_loss_curr = alpha_loss_total/(CONFIG.BATCH_SIZE)
            beta_loss_curr = beta_loss_total/(CONFIG.BATCH_SIZE)
            total_loss = alpha_loss_curr + beta_loss_curr
            print("Current f1 score: ", f1score_curr)
            print("Current em score: ", emscore_curr)
            print("Current alpha loss: ", alpha_loss_curr)
            print("Current beta loss: ", beta_loss_curr)
            f1score.append(f1score_curr)
            emscore.append(emscore_curr)
            loss.append(total_loss)

            #if(iteration % ((CONFIG.BATCH_SIZE)-1) == 0):
            print("Tested (",iteration,"/",len(padded_data),")")

        print("F1 mean: ", np.mean(f1score))
        print("EM mean: ", np.mean(emscore))
        print("Loss mean: ", np.mean(total_loss))

        f1_pickle_file = './RESULTS/testing_f1_means.pkl'
        em_pickle_file = './RESULTS/testing_em_means.pkl'
        loss_pickle_file = './RESULTS/testing_loss_means.pkl'
        if os.path.exists(f1_pickle_file):
            append_write = 'ab'  # append if already exists
        else:
            append_write = 'wb'  # make a new file if not
        with open(f1_pickle_file, append_write) as f:
            pickle.dump(np.mean(f1score), f, protocol=3)
        with open(em_pickle_file, append_write) as f:
            pickle.dump(np.mean(emscore), f, protocol=3)
        with open(loss_pickle_file, append_write) as f:
            pickle.dump(np.mean(total_loss), f, protocol=3)
