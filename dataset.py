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

        self.load_embeddings(sys.argv[1:])

    def generate_question_encoding(self, categories, word2index):
        print("Generating question encoding...")
        data = []
        skipped_count = 0
        for category in categories:
            for paragraph in category["paragraphs"]:
                split_context = tokenise(paragraph["context"])
                for qas in paragraph["qas"]:
                    # Translate character index to word index
                    answer = qas["answers"]
                    found = False
                    answer_index = 0

                    try: 
                        while not found:
                            split_answer = tokenise(answer[answer_index]["text"])

                            answer_start = next(KnuthMorrisPratt(split_context, split_answer))
                            if answer_start != None:
                                found = True
                            else:
                                answer_index += 1
                            
                        answer_end = answer_start + len(split_answer) - 1

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
        while ((words[-1] == '?') and (len(words) > 1)):
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
        self.index2word = defaultdict(lambda: '?', dict(zip(word2index.values(), word2index.keys())))

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
            assert data["version"] == "1.1"
            categories = data["data"]

        data = self.generate_question_encoding(categories, self.word2index) 

        # Pad questions and contexts
        pad_char = self.vocab_size-1
        padded_data, (max_length_question, max_length_context) = pad_data(data, pad_char)

        print("Loaded questions")
        return padded_data, max_length_question, max_length_context

if __name__ == '__main__':
    D = Dataset('data/dev.json', 'data/glove.840B.300d.txt')
    padded_data, index2embedding, max_length_question, max_length_context = D.load_data(sys.argv[1:])
    paragraph = {
        "context": "With 4:51 left in regulation, Carolina got the ball on their own 24-yard line with a chance to mount a game-winning drive, and soon faced 3rd-and-9. On the next play, Miller stripped the ball away from Newton, and after several players dove for it, it took a long bounce backwards and was recovered by Ward, who returned it five yards to the Panthers 4-yard line. Although several players dove into the pile to attempt to recover it, Newton did not and his lack of aggression later earned him heavy criticism. Meanwhile, Denver's offense was kept out of the end zone for three plays, but a holding penalty on cornerback Josh Norman gave the Broncos a new set of downs. Then Anderson scored on a 2-yard touchdown run and Manning completed a pass to Bennie Fowler for a 2-point conversion, giving Denver a 24\u201310 lead with 3:08 left and essentially putting the game away. Carolina had two more drives, but failed to get a first down on each one.", 
        "qas": [
            {
                "answers": [{"answer_start": 65, "text": "24"}, {"answer_start": 55, "text": "their own 24"}, {"answer_start": 65, "text": "24"}], 
                "question": "On what yard line did Carolina begin with 4:51 left in the game?", "id": "56beca913aeaaa14008c946d"
            }, 
            {
                "answers": [{"answer_start": 202, "text": "Newton"}, {"answer_start": 202, "text": "Newton"}, {"answer_start": 434, "text": "Newton"}],
                "question": "Who fumbled the ball on 3rd-and-9?", "id": "56beca913aeaaa14008c946e"
            },
            {
                "answers": [{"answer_start": 620, "text": "Josh Norman"}, {"answer_start": 620, "text": "Josh Norman"}, {"answer_start": 625, "text": "Norman"}],
                "question": "What Panther defender was called for holding on third down?", "id": "56beca913aeaaa14008c946f"
            },
        ]
    }
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

        print({
            "context": paragraph["context"],
            "question": qas["question"],
            "answer_start": answer_start,
            "answer_end": int(answer_start) + len(text_to_index(answer["text"], D.word2index)) - 1,
            "answer": paragraph["context"].split()[answer_start:int(answer_start) + len(text_to_index(answer["text"], D.word2index))],
            "answer_text":answer["text"]
        })