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

# custom imports
from preprocessing import text_to_index, load_embedding

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
        with open(PRESAVED_QUESTIONS_FILE, "wb") as question_file:
            pickle.dump(questions, question_file)
    else:
        print("Loading question encoding from file")
        with open(PRESAVED_QUESTIONS_FILE, "rb") as question_file:
            questions = pickle.load(question_file)

    print("Loaded test data")
    print(questions[0])
    print(".....")

    i = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(i)

if __name__ == "__main__":
    main(sys.argv[1:])