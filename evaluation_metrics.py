
import numpy as np
from collections import Counter
import re, string, sys
from dataset import Dataset
from config import CONFIG

D = Dataset(CONFIG.QUESTION_FILE, CONFIG.EMBEDDING_FILE)
padded_data, index2embedding, max_length_question, max_length_context = D.load_data(sys.argv[1:])

def squad_f1_score( prediction, ground_truth):
    """Method copied from the SQuAD Leaderboard: https://rajpurkar.github.io/SQuAD-explorer/"""
    prediction_tokens = squad_normalize_answer(prediction).split()
    ground_truth_tokens = squad_normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


""" Lower text and remove punctuation, articles and extra whitespace
Method copied from the SQuAD Leaderboard: https://rajpurkar.github.io/SQuAD-explorer/  """
def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)

def white_space_fix(text):
    return ' '.join(text.split())

def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

def lower(text):
    return text.lower()

def squad_normalize_answer(s):
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def indexToWord(indices):
    return D.index_to_text(indices)

def get_f1_from_tokens( yS, yE, ypS, ypE, batch_Xc):
    """
    Pass yS, yE, ypS and ypE to be indices.batch_Xc is the context indices

    This function doesn't compare the indices, but the tokens behind the indices. This is a bit more forgiving
    and it is the metric applied on the SQuAD leaderboard."""
    split_context = indexToWord(batch_Xc).split()
    ground_truth = ' '.join(split_context[yS:yE+1])
    prediction = ' '.join(split_context[ypS:ypE + 1])
    #prediction = index_list_to_string(batch_Xc[ypS:ypE + 1])
    f1 = squad_f1_score(prediction, ground_truth)
    return f1






