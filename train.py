# This file trains the neural network using the encoder and decoder.

import numpy as np
import tensorflow as tf
import pandas as pd
import json
import random

# custom imports
from preprocessing import text_to_vector

# open the training file 
TRAINING_FILE_NAME = 'data/dev.json'

with open(TRAINING_FILE_NAME, "r") as f:
    data = json.loads(f.read())
    assert data["version"] == "1.1"
    categories = data["data"]

questions = [
    {"question": "something", "answers":["something","else"]}
];

print("Loaded test data. Categories:")
for category in categories:
    print(category["title"] + ": " + str(len(category["paragraphs"])), end=", ")
    for paragraph in category["paragraphs"]:
        paragraph["context"] = text_to_vector(paragraph["context"])
        for qas in paragraph["qas"]:
            questions.append({
                "context": paragraph["context"],
                "question": text_to_vector(qas["question"]),
                "answer": random.choice(qas["answers"])["text"]
            })      
    
print(questions)


i = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(i)
