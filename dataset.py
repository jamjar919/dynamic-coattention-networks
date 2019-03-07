import random
import pickle
import pandas as pd
import json
import os
from collections import defaultdict  
import sys
from preprocessing import text_to_index, load_embedding, pad_data

class Dataset:

    def __init__(self, training_file, glove_file):
        self.TRAINING_FILE_NAME = training_file
        self.GLOVE_DATA_FILE = glove_file
        self.PRESAVED_QUESTIONS_FILE = 'generated/encoded_questions.pickle'
        self.PRESAVED_EMBEDDING_FILE = 'generated/embedding.pickle'

    def generate_question_encoding(self, categories, word2index):
        print("Generating question encoding...")
        data = []
        for category in categories:
            for paragraph in category["paragraphs"]:
                paragraph["context"] = paragraph["context"]
                split_context = paragraph["context"].split(" ")
                for qas in paragraph["qas"]:
                    # Translate character index to word index
                    answer = random.choice(qas["answers"])
                    i = 0
                    word_index = 0
                    while (i < answer["answer_start"]):
                        i += len(split_context[word_index]) + 1
                        word_index += 1
                    answer_start = word_index

                    data.append({
                        "context": text_to_index(paragraph["context"], word2index),
                        "question": text_to_index(qas["question"], word2index),
                        "answer_start": answer_start,
                        "answer_end": int(answer_start) + len(text_to_index(answer["text"], word2index)) - 1
                    })
        return data

    def generate_glove_vectors(self):
        print("Generating glove vectors...")
        return load_embedding(self.GLOVE_DATA_FILE)

    def load_if_cached_else_generate(self, filename, generator, REGENERATE_CACHE = False, beforesave = lambda x: x, then = lambda x: x):
        if (not os.path.isfile(filename)) or REGENERATE_CACHE:
            # generate
            result = generator()
            with open(filename, "wb") as f:
                pickle.dump(beforesave(result), f)
                print("Saved "+filename+" to disk.")

            return result
        else:
            print("Loading "+filename+" from disk.")
            with open(filename, "rb") as f:
                result = pickle.load(f)
            return then(result)


    def load_data(self, args):
        '''
            Function for loading the data and padding as required. 
            Pass command line option --regenerateEmbeddings to force write the embeddings to file
        '''
        REGENERATE_CACHE = '--regenerateEmbeddings' in args

        # read SQuAD data
        with open(self.TRAINING_FILE_NAME, "r") as f:
            data = json.loads(f.read())
            assert data["version"] == "1.1"
            categories = data["data"]

        word2index, index2embedding = self.load_if_cached_else_generate(
            self.PRESAVED_EMBEDDING_FILE,
            lambda: self.generate_glove_vectors(),
            REGENERATE_CACHE,
            beforesave = lambda x: (dict(x[0]), x[1]),
            then = lambda x: (defaultdict(lambda: len(x[0]), x[0]), x[1])
        )
        print("Loaded embeddings")
        vocab_size, embedding_dim = index2embedding.shape
        print("Vocab Size:"+str(vocab_size)+" Embedding Dim:"+str(embedding_dim))

        data = self.load_if_cached_else_generate(
            self.PRESAVED_QUESTIONS_FILE,
            lambda: self.generate_question_encoding(categories, word2index),
            REGENERATE_CACHE
        )
        print("Loaded questions")

        # Pad questions and contexts
        pad_char = vocab_size-1
        padded_data, (max_length_question, max_length_context) = pad_data(data, pad_char)
        return padded_data, index2embedding, max_length_question, max_length_context