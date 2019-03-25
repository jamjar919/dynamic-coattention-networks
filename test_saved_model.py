# This file runs the actual question answering program using our trained network
import sys
import numpy as np
import tensorflow as tf
from functools import reduce

# custom imports
from dataset import Dataset
from config import CONFIG
from build_model import get_batch
from evaluation_metrics import get_f1_from_tokens

print("Starting testing on dev file...")
D = Dataset('data/train.json', CONFIG.EMBEDDING_FILE)
padded_data, index2embedding, max_length_question, max_length_context = D.load_data(sys.argv[1:])
print("Loaded data")

#tf.reset_default_graph()
latest_checkpoint_path = './model/saved-23'
print("restoring from "+latest_checkpoint_path)
saver = tf.train.import_meta_graph(latest_checkpoint_path+'.meta')

#init = tf.global_variables_initializer()

with tf.Session() as sess:
    #sess.run(init)
    saver.restore(sess, latest_checkpoint_path)
    graph = tf.get_default_graph()
    answer_start_batch_predict = graph.get_tensor_by_name("answer_start:0")
    answer_end_batch_predict = graph.get_tensor_by_name("answer_end:0")
    question_batch_placeholder = graph.get_tensor_by_name("question_batch_ph:0")
    context_batch_placeholder = graph.get_tensor_by_name("context_batch_ph:0")
    embedding = graph.get_tensor_by_name("embedding_ph:0")
    dropout_keep_rate = graph.get_tensor_by_name("dropout_keep_ph:0")
    #loss  = graph.get_tensor_by_name("loss_to_optimize:0")

    f1score = []

    print("SESSION INITIALIZED")
    for iteration in range(0, len(padded_data) - CONFIG.BATCH_SIZE, CONFIG.BATCH_SIZE):
        # running on an example batch to debug encoder
        batch = padded_data[iteration:(iteration + CONFIG.BATCH_SIZE)]
        question_batch, context_batch, answer_start_batch_actual, answer_end_batch_actual = get_batch(batch, CONFIG.BATCH_SIZE, max_length_question, max_length_context)
        
        estimated_start_index, estimated_end_index = sess.run([answer_start_batch_predict, answer_end_batch_predict], feed_dict={
            question_batch_placeholder: question_batch,
            context_batch_placeholder: context_batch,
            embedding: index2embedding,
            dropout_keep_rate: 1
        })
        #print("Loss: ", np.mean(loss))
        print("estimated start index: ", estimated_start_index)
        print("estimated end index: ", estimated_end_index)
        print(estimated_start_index.shape)

        f1 = 0            
        for i in range(len(estimated_end_index)):
            f1 += get_f1_from_tokens(
                answer_start_batch_actual[i], 
                answer_end_batch_actual[i],
                estimated_start_index[i], estimated_end_index[i],
                context_batch[i],
                D
            )
        f1score_curr = f1/len(estimated_end_index)
        print("Current f1 score: ", f1score_curr)
        f1score.append(f1score_curr)

        #if(iteration % ((CONFIG.BATCH_SIZE)-1) == 0):
        print("Tested (",iteration,"/",len(padded_data),")")

    print("F1 mean: ", np.mean(f1score))