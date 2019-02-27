# preprocessing of the data

import tensorflow as tf
import pandas as pd 
import numpy as np
import csv
import sys
from collections import defaultdict  

def pad_to(sequence, length, char):
    if len(sequence) >= length:
        return sequence
    else:
        sequence.append(char)
        return pad_to(sequence, length, char)


def word_to_index(w, word2index):
    try:
        result = word2index[w]
        return result
    except KeyError:
        return len(word2index) - 1 # defined to be all zeros

def text_to_index(text, word2index):
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
    return list(map(lambda tok: word_to_index(tok, word2index), tokens));

#https://github.com/guillaume-chevalier/GloVe-as-a-TensorFlow-Embedding-Layer/blob/master/GloVe-as-TensorFlow-Embedding-Tutorial.ipynb
def load_embedding(glove):
    word_to_index_dict = dict()
    index_to_embedding_array = []
    
    with open(glove, 'r', encoding="utf-8") as glove_file:
        for (i, line) in enumerate(glove_file):
            split = line.split(' ')
            
            word = split[0]
            
            representation = split[1:]
            representation = np.array(
                [float(val) for val in representation]
            )
            
            word_to_index_dict[word] = i
            index_to_embedding_array.append(representation)

    _WORD_NOT_FOUND = [0.0]* len(representation)  # Empty representation for unknown words.
    _LAST_INDEX = i + 1

    word_to_index_dict = defaultdict(lambda: _LAST_INDEX, word_to_index_dict)
    index_to_embedding_array = np.array(index_to_embedding_array + [_WORD_NOT_FOUND])
    return word_to_index_dict, index_to_embedding_array