# preprocessing of the data

import tensorflow as tf
import pandas as pd 
import numpy as np
import csv

GLOVE_DATA_FILE = 'data/glove.6B.300d.txt'
words = pd.read_csv(GLOVE_DATA_FILE, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

print("read words")

def vec(w):
  return words.loc[w].to_numpy()

def text_to_vector(text):
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
    vector = np.zeros((len(tokens), 300), dtype=np.float32)
    for i in range(0, len(tokens)):
        try:
            vector[i] = vec(tokens[i])
        except KeyError: 
            pass;
    
    print(vector);
    print(vector.shape)
    return vector;
