# This file trains the neural network using the encoder and decoder.
import nltk
import sys
import wikipedia
from dataset import Dataset
from config import CONFIG
import tensorflow as tf
import numpy as np
from preprocessing import answer_span_to_indices
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

question_asked = input("Enter a \'wh\' question, for example: Who is Sachin Ramesh Tendulkar?\n")
text = nltk.word_tokenize(question_asked)
processed_pos = nltk.pos_tag(text)
text_to_search = ''
print(processed_pos)
for index in range(len(processed_pos)):
    if( "VB" in processed_pos[index][1]):
        text_to_search = ' '.join(question_asked.split(' ')[index+1:])
        break


print("the test to search " + text_to_search)
summary = wikipedia.summary(text_to_search)

print(len(summary))
context = ' '.join(summary.split()[:CONFIG.MAX_CONTEXT_LENGTH - 2])
print("searching in  " + context)
D = Dataset( CONFIG.EMBEDDING_FILE)
index2embedding = D.index2embedding
question_encoding, context_encoding = D.encode_single_question(question_asked, context, CONFIG.MAX_QUESTION_LENGTH, CONFIG.MAX_CONTEXT_LENGTH)

# random_question = np.random.choice(padded_data)
# print(len(random_question))
# print(len(padded_data))
# print(random_question)


embedding_dimension = 300
init = tf.global_variables_initializer()

latest_checkpoint_path = './model/saved-7'
print("restoring from "+latest_checkpoint_path)
saver = tf.train.import_meta_graph(latest_checkpoint_path+'.meta')

config = tf.ConfigProto()
if '--noGPU' in sys.argv[1:]:
    print("Not using the GPU...")
    config = tf.ConfigProto(device_count = {'GPU': 0})

with tf.Session(config=config) as sess:
    sess.run(init)
    saver.restore(sess, latest_checkpoint_path)
    graph = tf.get_default_graph()
    s = graph.get_tensor_by_name("answer_start:0")
    e = graph.get_tensor_by_name("answer_end:0")
    question_batch_placeholder = graph.get_tensor_by_name("question_batch_ph:0")
    context_batch_placeholder = graph.get_tensor_by_name("context_batch_ph:0")
    embedding = graph.get_tensor_by_name("embedding_ph:0")
    dropout_keep_rate = graph.get_tensor_by_name("dropout_keep_ph:0")

    question = np.array([question_encoding] * CONFIG.BATCH_SIZE, dtype = np.int32)
    context = np.array([context_encoding] * CONFIG.BATCH_SIZE, dtype = np.int32)

    s_result, e_result = sess.run([s, e], feed_dict = {
        question_batch_placeholder : question,
        context_batch_placeholder : context,
        embedding: index2embedding,
        dropout_keep_rate: 1
    })

    print("****8")
    print(s_result)

    print("****8")
    print(e_result)
    s_result = int(np.median(s_result))
    e_result = int(np.median(e_result))
    answer = answer_span_to_indices(s_result, e_result, context_encoding)

    print()
    print(D.index_to_text(question_encoding), " -> ", D.index_to_text(answer))