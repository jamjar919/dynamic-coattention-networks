import random
import pickle
import pandas as pd
import json
import os
from collections import defaultdict  
import sys
import tensorflow as tf

from preprocessing import text_to_index, load_embedding, pad_data, KnuthMorrisPratt, tokenise

class Dataset:

    def __init__(self, glove_file):
        self.GLOVE_DATA_FILE = glove_file
        self.PRESAVED_EMBEDDING_FILE_NAME = 'embedding.pickle'
        self.PRESAVED_DIR = 'generated/'

        self.word2index = None
        self.vocab_size = 0
        self.embedding_dim = 0
        self.unknown_word = "<UNKNOWN>"

        self.index2embedding = self.load_embeddings(sys.argv[1:])

    def generate_question_encoding(self, categories, word2index, version="1.1"):
        print("Generating question encoding...")
        data = []
        skipped_count = 0
        for category in categories:
            for paragraph in category["paragraphs"]:
                split_context = tokenise(paragraph["context"])
                for qas in paragraph["qas"]:
                    # Translate character index to word index
                    answers = qas["answers"]
                    
                    found = False
                    answer_index = 0

                    try: 
                        if (len(answers) > 0):
                            while not found:
                                split_answer = tokenise(answers[answer_index]["text"])

                                answer_start = next(KnuthMorrisPratt(split_context, split_answer))
                                if answer_start != None:
                                    found = True
                                else:
                                    answer_index += 1
                                
                            answer_end = answer_start + len(split_answer) - 1
                        elif (version == "v2.0") and (qas["is_impossible"]):
                            answer_start = -1
                            answer_end = -1

                        data.append({
                            "context": text_to_index(paragraph["context"], word2index),
                            "question": text_to_index(qas["question"], word2index),
                            "answer_start": answer_start,
                            "answer_end": answer_end
                        })
                    except IndexError:
                        skipped_count += 1
        print("Skipped encoding",skipped_count,"/",len(data) + skipped_count,"questions because couldn't find the answer in the text")
        return data

    def generate_glove_vectors(self):
        print("Generating glove vectors...")
        return load_embedding(self.GLOVE_DATA_FILE)

    def load_if_cached_else_generate(self, filename, generator, REGENERATE_CACHE = False, beforesave = lambda x: x, then = lambda x: x):
        if (not os.path.isfile(filename)) or REGENERATE_CACHE:
            # generate
            result = generator()
            with open(filename, "wb") as f:
                pickle.dump(beforesave(result), f, protocol=4)
                print("Saved "+filename+" to disk.")

            return result
        else:
            print("Loading "+filename+" from disk.")
            with open(filename, "rb") as f:
                result = pickle.load(f)
            return then(result)

    def index_to_text(self, indexes):
        words = list(map(lambda index: self.index2word[index], indexes))
        while ((words[-1] == self.unknown_word) and (len(words) > 1)):
            words = words[:-1]
        return ' '.join(words)

    def load_embeddings(self, args):
        '''
            Function for loading the data and padding as required. 
            Pass command line option --regenerateEmbeddings to force write the embeddings to file
        '''
        REGENERATE_CACHE = '--regenerateEmbeddings' in args

        word2index, index2embedding = self.load_if_cached_else_generate(
            self.PRESAVED_DIR + self.PRESAVED_EMBEDDING_FILE_NAME,
            lambda: self.generate_glove_vectors(),
            REGENERATE_CACHE,
            beforesave = lambda x: (dict(x[0]), x[1]),
            then = lambda x: (defaultdict(lambda: len(x[0]), x[0]), x[1])
        )
        self.word2index = word2index
        self.index2word = defaultdict(lambda: self.unknown_word, dict(zip(word2index.values(), word2index.keys())))

        print("Loaded embeddings")
        self.vocab_size, self.embedding_dim = index2embedding.shape
        print("Vocab Size:"+str(self.vocab_size)+" Embedding Dim:"+str(self.embedding_dim))

        return index2embedding

    def load_questions(self, question_file):
        if (self.word2index == None):
            raise RuntimeError("Load the embedding file first")

        # read SQuAD data
        with open(question_file, "r") as f:
            data = json.loads(f.read())
            version = data["version"]
            categories = data["data"]

        print("Question version is",version)

        data = self.generate_question_encoding(categories, self.word2index, version) 

        # Pad questions and contexts
        pad_char = self.vocab_size-1
        padded_data, (max_length_question, max_length_context) = pad_data(data, pad_char)

        print("Loaded questions")
        return (padded_data, (max_length_question, max_length_context))

if __name__ == '__main__':
    from config import CONFIG
    D = Dataset('data/glove.840B.300d.txt')
    index2embedding = D.index2embedding
    padded_data, (max_length_question, max_length_context) = D.load_questions(CONFIG.QUESTION_FILE)

    print(D.index_to_text(padded_data[0]["context"]))
