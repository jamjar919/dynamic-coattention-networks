# This file trains the neural network using the encoder and decoder.
import sys
import numpy as np
import tensorflow as tf
import pickle
from functools import reduce
import os
# custom imports
from encoder import encoder
from decoder import decoder
from preprocessing import answer_span_to_indices
from dataset import Dataset
from config import CONFIG
from evaluation_metrics import get_f1_from_tokens, get_exact_match_from_tokens
from build_model import build_model, get_feed_dict, get_batch

tensorboard_filepath = '.'

D = Dataset(CONFIG.EMBEDDING_FILE)
index2embedding = D.index2embedding
padded_data, (max_length_question, max_length_context) = D.load_questions(CONFIG.QUESTION_FILE)
print("Loaded data")

tf.reset_default_graph()
embedding = tf.placeholder(shape = [len(index2embedding), CONFIG.EMBEDDING_DIMENSION], dtype=tf.float32, name='embedding_ph')
train_op, loss, s, e  = build_model(embedding)

# Blank csv file
open('./results/training_loss_per_batch.csv', 'w').close()

# Train now
saver = tf.train.Saver(max_to_keep = CONFIG.MAX_EPOCHS) 
init = tf.global_variables_initializer()

config = tf.ConfigProto()
if '--noGPU' in sys.argv[1:]:
    print("Not using the GPU...")
    config = tf.ConfigProto(device_count = {'GPU': 0})

with tf.Session(config=config) as sess:
    sess.run(init)
    print("SESSION INITIALIZED")
    dataset_size = len(padded_data)
    padded_data = np.array(padded_data)
    
    #print("PADDED DATA SHAPE: ", padded_data.shape)
    padded_data_train = padded_data[0:(int) (CONFIG.TRAIN_PERCENTAGE*padded_data.shape[0])]
    padded_data_validation = padded_data[(int) (CONFIG.TRAIN_PERCENTAGE*padded_data.shape[0]):]
    #np.random.shuffle(padded_data_train)
    #np.random.shuffle(padded_data_validation)

    print("LEN PADDED DATA TRAIN: ", len(padded_data_train))
    loss_means = []
    val_loss_means = []
    val_f1_means = []
    val_em_means = []
    for epoch in range(1):
        print("Epoch # : ", epoch + 1)
        losses = []
        # Shuffle the data between epochs
        #np.random.shuffle(padded_data_train)
        for iteration in range(0, 70): #len(padded_data_train) - CONFIG.BATCH_SIZE, CONFIG.BATCH_SIZE):
            batch = padded_data_train[0: 0+CONFIG.BATCH_SIZE] #[iteration:iteration + CONFIG.BATCH_SIZE]
            question_batch, context_batch, answer_start_batch, answer_end_batch = get_batch(batch, CONFIG.BATCH_SIZE, max_length_question, max_length_context)
            
            if (iteration == 0):
                print("First context: ", D.index_to_text(context_batch[0]))
            print("First question: ", D.index_to_text(question_batch[0]))
            print("Answer indices : ", answer_start_batch[0], answer_end_batch[0])
            answer = answer_span_to_indices(answer_start_batch[0], answer_end_batch[0], context_batch[0])
            print("First answer label: ", D.index_to_text(answer))

            _ , loss_val = sess.run([train_op, loss],feed_dict = get_feed_dict(question_batch,context_batch,answer_start_batch,answer_end_batch, CONFIG.DROPOUT_KEEP_PROB, index2embedding))

            print("Training Loss: ", np.mean(loss_val), "(",iteration,"/",len(padded_data_train),")")

        estimated_start_index, estimated_end_index, loss_valid = sess.run([s, e, loss],
        feed_dict = get_feed_dict(question_batch,context_batch,answer_start_batch,answer_end_batch, 1.0, index2embedding))
        
        est_answer = answer_span_to_indices(estimated_start_index[0], estimated_end_index[0], context_batch[0])
        print("Predicted answer 1: ", D.index_to_text(est_answer))
        est_answer = answer_span_to_indices(estimated_start_index[1], estimated_end_index[1], context_batch[1])
        print("Predicted answer 2: ", D.index_to_text(est_answer))
        
        print("loss_validation : ", np.mean(loss_valid))
        f1 = 0
        em = 0
        for i in range(len(estimated_end_index)):
            #print("start actual, end actual, start pred, end pred: ", answer_start_batch_actual[i], answer_end_batch_actual[i], estimated_start_index[i], estimated_end_index[i])
            f1 += get_f1_from_tokens(answer_start_batch[i], answer_end_batch[i],
                                estimated_start_index[i], estimated_end_index[i],
                                context_batch[i], D )
            em += get_exact_match_from_tokens(answer_start_batch[i], answer_end_batch[i],
                                estimated_start_index[i], estimated_end_index[i],
                                context_batch[i], D )
        print("F1: ", f1/len(estimated_end_index))
        print("EM match: ", em / len(estimated_end_index) )
            #est_answer = answer_span_to_indices(shat[0], ehat[0], context_batch[0])
            #print("Predicted answer: ", D.index_to_text(est_answer))
            #if(iteration % ((CONFIG.BATCH_SIZE)-1) == 0):
            
            

            #losses.append(loss_val_mean.item())
        
        '''
        mean_epoch_loss = np.mean(np.array(losses))
        loss_means.append(mean_epoch_loss)
        print("Mean epoch loss: ", mean_epoch_loss)
        f1score = []
        emscore = []
        validation_losses = []

        #validation starting
        for counter in range(0, 10): #len(padded_data_validation) - CONFIG.BATCH_SIZE, CONFIG.BATCH_SIZE):
            batch = padded_data_validation[0: 0+CONFIG.BATCH_SIZE] #counter:(counter + CONFIG.BATCH_SIZE)]
            question_batch_validation, context_batch_validation, answer_start_batch_actual, answer_end_batch_actual = get_batch(batch, CONFIG.BATCH_SIZE, max_length_question, max_length_context)

            estimated_start_index, estimated_end_index, loss_validation = sess.run([s, e, loss],
            get_feed_dict(question_batch_validation,context_batch_validation,answer_start_batch_actual,answer_end_batch_actual, 1.0, index2embedding)
            )

            validation_losses.append(np.mean(loss_validation))
            f1 = 0
            em = 0
            for i in range(len(estimated_end_index)):
                #print("start actual, end actual, start pred, end pred: ", answer_start_batch_actual[i], answer_end_batch_actual[i], estimated_start_index[i], estimated_end_index[i])
                f1 += get_f1_from_tokens(answer_start_batch_actual[i], answer_end_batch_actual[i],
                                   estimated_start_index[i], estimated_end_index[i],
                                   context_batch_validation[i], D )
                em += get_exact_match_from_tokens(answer_start_batch_actual[i], answer_end_batch_actual[i],
                                   estimated_start_index[i], estimated_end_index[i],
                                   context_batch_validation[i], D )

            f1score.append(f1 / len(estimated_end_index))
            emscore.append(em / len(estimated_end_index))
            #print("f1 score: ", f1/len(estimated_end_index))
  
        print("F1 mean on validation: ", np.mean(f1score))
        print("EM mean on validation: ", np.mean(emscore))
        print("Mean validation loss on epoch: ", np.mean(validation_losses))
        val_loss_means.append(np.mean(validation_losses))
        val_f1_means.append(np.mean(f1score))
        val_em_means.append(np.mean(emscore))

        with open('./results/validation_loss_means.pkl', 'wb') as f:
            pickle.dump(val_loss_means, f, protocol=3)
        with open('./results/validation_f1_means.pkl', 'wb') as f:
            pickle.dump(val_f1_means, f, protocol=3)
        with open('./results/validation_em_means.pkl', 'wb') as f:
            pickle.dump(val_em_means, f, protocol=3)
        with open('./results/training_loss_means.pkl', 'wb') as f:
            pickle.dump(loss_means, f, protocol = 3)
        with open('./results/training_loss_per_batch.csv', 'a+') as f:
            f.write(','.join(list(map(lambda x: str(x), losses))) + '\n')

        saver.save(sess, './model/saved', global_step=epoch)

        '''