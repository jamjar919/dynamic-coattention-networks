import numpy as np
from collections import Counter
import re, string, sys
from preprocessing.dataset import Dataset
from network.config import CONFIG
import collections

def indexToWord(indices, D):
    return D.index_to_text(indices)

def get_f1_from_tokens( actualStartIndex, actualEndIndex, predictedStartIndex, predictedEndIndex, batch_Xc, D):
    split_context = indexToWord(batch_Xc, D).split()
    ground_truth = ' '.join(split_context[actualStartIndex:actualEndIndex+1])
    prediction = ' '.join(split_context[predictedStartIndex:predictedEndIndex + 1])
    #prediction = index_list_to_string(batch_Xc[predictedStartIndex:predictedEndIndex + 1])
    f1 = compute_f1(ground_truth, prediction)
    return f1

def get_exact_match_from_tokens(actualStartIndex, actualEndIndex, predictedStartIndex, predictedEndIndex, batch_Xc, D):
    em = 0
    #TODO: Pull ou the code from this and get_f1_from_tokens fn
    split_context = indexToWord(batch_Xc, D).split()
    ground_truth = ' '.join(split_context[actualStartIndex:actualEndIndex+1])
    prediction = ' '.join(split_context[predictedStartIndex:predictedEndIndex + 1])
    em += compute_exact(ground_truth, prediction)
    return em

# Methods copied from SQuAD leaderboard
def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

if __name__ == "__main__":
    D = Dataset(CONFIG.EMBEDDING_FILE)
    index2embedding = D.index2embedding
    padded_data, (max_length_question, max_length_context) = D.load_questions(CONFIG.QUESTION_FILE)

    ty = np.array([1, 5, 4, 7, 8, 4, 3, 4, 6, 7, 7, 4])
    print(get_f1_from_tokens(5, 8, 4, 7,ty, D))
    print(get_f1_from_tokens(5, 8, 5, 8,ty, D))
    print(get_f1_from_tokens(5, 8, 1, 3,ty, D))
    print(get_f1_from_tokens(5, 8, 6, 8,ty, D))
    print(get_f1_from_tokens(5, 8, 1, 8,ty, D))
    # print(get_f1_from_tokens(5, 8, 8, 9,ty, D))
    print(get_exact_match_from_tokens(5, 8, 4, 7,ty, D))
    print(get_exact_match_from_tokens(1, 4, 1, 4, ty, D))








