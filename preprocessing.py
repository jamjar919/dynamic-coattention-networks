# preprocessing of the data

import tensorflow as tf
import pandas as pd 
import numpy as np
import csv
import sys
from collections import defaultdict  
from functools import reduce
import re
import string
from config import CONFIG

# https://code.activestate.com/recipes/117214/
def KnuthMorrisPratt(text, pattern):

    '''Yields all starting positions of copies of the pattern in the text.
Calling conventions are similar to string.find, but its arguments can be
lists or iterators, not just strings, it returns all matches, not just
the first one, and it does not need the whole text in memory at once.
Whenever it yields, it will have read the text exactly up to and including
the match that caused the yield.'''

    # allow indexing into pattern and protect against change during yield
    pattern = list(pattern)

    # build table of shift amounts
    shifts = [1] * (len(pattern) + 1)
    shift = 1
    for pos in range(len(pattern)):
        while shift <= pos and pattern[pos] != pattern[pos-shift]:
            shift += shifts[pos-shift]
        shifts[pos+1] = shift

    # do the actual search
    startPos = 0
    matchLen = 0
    for c in text:
        while matchLen == len(pattern) or \
              matchLen >= 0 and pattern[matchLen] != c:
            startPos += shifts[matchLen]
            matchLen -= shifts[matchLen]
        matchLen += 1
        if matchLen == len(pattern):
            yield startPos

    yield None

def pad_data(data, pad_char):
    max_length_question = len(reduce(lambda a, b: b if len(a) < len(b) else a, list(map(lambda a: a["question"], data))))
    max_length_context = len(reduce(lambda a, b: b if len(a) < len(b) else a, list(map(lambda a: a["context"], data))))

    if CONFIG.MAX_QUESTION_LENGTH != None:
        max_length_question = CONFIG.MAX_QUESTION_LENGTH
    
    if CONFIG.MAX_CONTEXT_LENGTH != None:
        max_length_context = CONFIG.MAX_CONTEXT_LENGTH

    padded_data = []

    for q in data:
        question, question_mask = pad_to(q["question"], max_length_question, pad_char)
        context, context_mask = pad_to(q["context"], max_length_context, pad_char)
        if (q["answer_end"] <= max_length_context):
            padded_data.append({
                "question": question,
                "context": context,
                "question_mask": question_mask,
                "context_mask": context_mask,
                "answer_start": q["answer_start"],
                "answer_end": q["answer_end"],
                "all_answers": q["all_answers"]
            })
    return padded_data, (max_length_question, max_length_context)

def pad_to(sequence, length, char):
    if len(sequence) > length:
        return (sequence[0:length], [True] * length)
    if len(sequence) == length:
        return (sequence, [True] * length)
    else:
        mask = [True] * len(sequence)
        while len(sequence) != length:
            mask.append(False)
            sequence.append(char)
        return sequence, mask

def word_to_index(w, word2index):
    try:
        result = word2index[w]
        return result
    except KeyError:
        return len(word2index) - 1 # defined to be all zeros

def tokenise(text):
    # Replace annoying unicode with a space
    text = re.sub(r'[^\x00-\x7F]+',' ', text)
    # The following replacements are suggested in the paper
    # BidAF (Seo et al., 2016)
    text = text.replace("''", '" ')
    text = text.replace("``", '" ')

    # Space out punctuation
    space_list = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~"
    text = text.translate(str.maketrans({key: " {0} ".format(key) for key in space_list}))

    # space out singlequotes a bit better (and like stanford)
    text = text.replace("'", " '")

    return tf.keras.preprocessing.text.text_to_word_sequence(text, lower=False, split=' ', filters='\t\n')

def text_to_index(text, word2index):
    tokens = tokenise(text)
    return list(map(lambda tok: word_to_index(tok, word2index), tokens))

def answer_span_to_indices(start, end, context_indexes):
    r = np.arange(int(start), int(end) + 1)
    return list(map(lambda index: context_indexes[index], r))

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