
import numpy as np
from collections import Counter
import re, string, sys
from dataset import Dataset
from config import CONFIG

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

def indexToWord(indices, D):
    return D.index_to_text(indices)

def get_f1_from_tokens( yS, yE, ypS, ypE, batch_Xc, D):
    """
    Pass yS, yE, ypS and ypE to be indices. batch_Xc is the context indices

    This function doesn't compare the indices, but the tokens behind the indices. This is a bit more forgiving
    and it is the metric applied on the SQuAD leaderboard."""
    split_context = indexToWord(batch_Xc, D).split()
    ground_truth = ' '.join(split_context[yS:yE+1])
    prediction = ' '.join(split_context[ypS:ypE + 1])
    #prediction = index_list_to_string(batch_Xc[ypS:ypE + 1])
    f1 = squad_f1_score(prediction, ground_truth)
    return f1

def squad_exact_match_score( prediction, ground_truth):
    """Method copied from the SQuAD Leaderboard: https://rajpurkar.github.io/SQuAD-explorer/"""
    return (squad_normalize_answer(prediction) == squad_normalize_answer(ground_truth))


def get_exact_match_from_tokens(yS, yE, ypS, ypE, batch_Xc, D):
    """This function doesn't compare the indices, but the tokens behind the indices. This is a bit more forgiving
    and it is the metric applied on the SQuAD leaderboard"""

    em = 0
    #TODO: Pull ou the code from this and get_f1_from_tokens fn
    split_context = indexToWord(batch_Xc, D).split()
    ground_truth = ' '.join(split_context[yS:yE+1])
    prediction = ' '.join(split_context[ypS:ypE + 1])
    em += squad_exact_match_score(prediction, ground_truth)
    return em


if __name__ == "__main__":
    D = Dataset(CONFIG.QUESTION_FILE, CONFIG.EMBEDDING_FILE)
    padded_data, index2embedding, max_length_question, max_length_context = D.load_data(sys.argv[1:])
    ty = np.array([1, 5, 4, 7, 8, 4, 3, 4, 6, 7, 7, 4])
    print(get_f1_from_tokens(5, 8, 4, 7,ty, D))
    print(get_f1_from_tokens(5, 8, 5, 8,ty, D))
    print(get_f1_from_tokens(5, 8, 1, 3,ty, D))
    print(get_f1_from_tokens(5, 8, 6, 8,ty, D))
    print(get_f1_from_tokens(5, 8, 1, 8,ty, D))
    # print(get_f1_from_tokens(5, 8, 8, 9,ty, D))
    print(get_exact_match_from_tokens(5, 8, 4, 7,ty, D))
    print(get_exact_match_from_tokens(1, 4, 1, 4, ty, D))







