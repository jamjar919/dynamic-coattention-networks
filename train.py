# This file trains the neural network using the encoder and decoder.

import numpy as np
import tensorflow as tf
import pandas as pd
import json
import random

# custom imports
from preprocessing import text_to_index, load_embedding

# open the training file 
TRAINING_FILE_NAME = 'data/dev.json'
GLOVE_DATA_FILE = 'data/glove.6B.300d.txt'

with open(TRAINING_FILE_NAME, "r") as f:
    data = json.loads(f.read())
    assert data["version"] == "1.1"
    categories = data["data"]

questions = [];

# load GLoVE vectors 
word2index, index2embedding = load_embedding(GLOVE_DATA_FILE)
vocab_size, embedding_dim = index2embedding.shape
embeddings = tf.constant(index2embedding, dtype=tf.float32)
print("Loaded embeddings")
print("Vocab Size:"+str(vocab_size)+" Embedding Dim:"+str(embedding_dim))

for category in categories:
    for paragraph in category["paragraphs"]:
        paragraph["context"] = paragraph["context"]
        for qas in paragraph["qas"]:
            questions.append({
                "context": text_to_index(paragraph["context"], word2index),
                "question": text_to_index(qas["question"], word2index),
                "answer": text_to_index(random.choice(qas["answers"])["text"], word2index)
            })

print("Loaded test data")
print(questions)

i = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(i)
