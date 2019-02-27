# This file trains the neural network using the encoder and decoder.

import numpy as np
import tensorflow as tf
import pandas as pd
import json
import random
import pickle
import os
from collections import defaultdict  
import sys
from functools import reduce


# custom imports
from preprocessing import text_to_index, load_embedding, pad_to
from encoder import encoder

# open the training file 
TRAINING_FILE_NAME = 'data/dev.json'
GLOVE_DATA_FILE = 'data/glove.6B.300d.txt'
PRESAVED_QUESTIONS_FILE = 'generated/encoded_questions.pickle'
PRESAVED_EMBEDDING_FILE = 'generated/embedding.pickle'



def main(args):
    '''
        Main function for training the network. 
        Pass command line option --regenerateEmbeddings to force write the embeddings to file
    '''
    REGENERATE_CACHE = '--regenerateEmbeddings' in args

    # read SQuAD data
    with open(TRAINING_FILE_NAME, "r") as f:
        data = json.loads(f.read())
        assert data["version"] == "1.1"
        categories = data["data"]

    questions = [];

    # load GLoVE vectors
    if (not os.path.isfile(PRESAVED_EMBEDDING_FILE)) or REGENERATE_CACHE:
        print("Generating embedding...")
        word2index, index2embedding = load_embedding(GLOVE_DATA_FILE)
        with open(PRESAVED_EMBEDDING_FILE, "wb") as embedding_file:
            pickle.dump((dict(word2index), index2embedding), embedding_file)
    else:
        print("Loading embedding from file")
        with open(PRESAVED_EMBEDDING_FILE, "rb") as embedding_file:
            word2index, index2embedding = pickle.load(embedding_file)
        word2index = defaultdict(lambda: len(word2index), word2index)

    print("Loaded embeddings")
    vocab_size, embedding_dim = index2embedding.shape
    embeddings = tf.constant(index2embedding, dtype=tf.float32)
    print("Vocab Size:"+str(vocab_size)+" Embedding Dim:"+str(embedding_dim))

    # Generate question encoding
    if (not os.path.isfile(PRESAVED_QUESTIONS_FILE)) or REGENERATE_CACHE:
        print("Generating question encoding...")
        for category in categories:
            for paragraph in category["paragraphs"]:
                paragraph["context"] = paragraph["context"]
                for qas in paragraph["qas"]:
                    questions.append({
                        "context": text_to_index(paragraph["context"], word2index),
                        "question": text_to_index(qas["question"], word2index),
                        "answer": text_to_index(random.choice(qas["answers"])["text"], word2index)
                    })

        max_length_questions = len(reduce(lambda a, b: b if len(a) < len(b) else a, list(map(lambda a: a["question"], questions))))
        max_length_context = len(reduce(lambda a, b: b if len(a) < len(b) else a, list(map(lambda a: a["context"], questions)))

        with open(PRESAVED_QUESTIONS_FILE, "wb") as question_file:
            pickle.dump(questions, question_file)
    else:
        print("Loading question encoding from file")
        with open(PRESAVED_QUESTIONS_FILE, "rb") as question_file:
            questions = pickle.load(question_file)

    print("Loaded test data")
    print(questions[0])
    print(".....")

    question_batch = list(map(lambda q: pad_to(q, max_length_questions, pad_char), question_batch))
    context_batch = list(map(lambda c: pad_to(c, max_length_context, pad_char), context_batch))


    tf.reset_default_graph()
    i = tf.global_variables_initializer()

    question = tf.placeholder(dtype=tf.int32,shape=[None,max_l_question])
    context = tf.placeholder(dtype=tf.int32,shape=[None,max_l_context])
    encoder_states = encoder(question,context,embeddings)

    with tf.Session() as sess:
        sess.run(i)
        batch_size = 10
        counter = 0
        pad_char = vocab_size - 1
        for epochs in range(len(questions) // batch_size):
            batch = questions[counter:(counter+batch_size)]

            question_batch = list(map(lambda qas: (qas["question"]), batch))
            context_batch = list(map(lambda qas: (qas["context"]), batch))

            counter = counter + batch_size
            for i in range(0, batch_size):
                question = tf.constant(question_batch[i], shape=(max_length_questions, 300))
                context = tf.constant(context_batch[i], shape=(max_length_context, 300))
                encoder_states = encoder(question, context, embeddings)

if __name__ == "__main__":
    main(sys.argv[1:])