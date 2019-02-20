# preprocessing of the data

import tensorflow as tf
import pandas as pd 
import csv

GLOVE_DATA_FILE = 'data/glove.6B.300d.txt'
words = pd.read_table(GLOVE_DATA_FILE, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

def vec(w):
  return words.loc[w].as_matrix()

def text_to_vector(text):
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
    for tok in tokens: 
        return vec(tok)
    return tokens;
