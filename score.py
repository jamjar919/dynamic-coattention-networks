import numpy as np

class Score :

    def __init__(self):
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0 
        self.false_negatives = 0
        self.precision = 0
        self.recall = 0 
        self.accuracy = 0
        self.F1 = 0

    def update(self, predicted_labels, actual_labels):
        for i in range(len(predicted_labels)):
            if actual_labels[i] == 0 :
                if predicted_labels[i] == 0 :
                    self.true_negatives+=1
                elif predicted_labels[i] == 1:
                    self.false_positives+=1
            elif actual_labels[i] == 1 :
                if predicted_labels[i] == 0:
                    self.false_negatives+=1
                elif predicted_labels[i] == 1:
                    self.true_positives+=1
        self.update_stats()

    def safe_div(self,n,d) :
        if d == 0:
            return 0
        return n/d
    
    def update_stats(self) :
        self.precision = self.safe_div(self.true_positives,self.true_positives + self.false_positives)
        self.recall = self.safe_div(self.true_positives, self.true_positives+self.false_negatives)
        self.F1 = self.safe_div(2 * (self.precision + self.recall), (self.precision * self.recall))
        self.accuracy = self.safe_div(self.true_positives + self.true_negatives , self.true_positives + 
            self.true_negatives + self.false_positives + self.false_negatives)
    
    def reset(self):
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0 
        self.false_negatives = 0
        self.precision = 0
        self.recall = 0 
        self.accuracy = 0
        self.F1 = 0

    def print_stats(self):
        print(f"tp: {self.true_positives}, tn: {self.true_negatives}, fp: {self.false_positives}, fn: {self.false_negatives}")
        print(f"precision = {self.precision}, recall = {self.recall}, F1 = {self.F1}, accuracy = {self.accuracy}")

if __name__ == '__main__':
    score = Score()
    score.print_stats()
    score.update([1,1,1,1],[1,1,1,1])
    score.print_stats()
    # test with np array
    score.update(np.array([0,0,0,0]),[0,0,0,0])
    score.print_stats()
    score.update([1,1,0,0],[0,0,1,1])
    score.print_stats()
    score.update([1],[0])
    score.print_stats()
    score.update([0],[1])
    score.print_stats()

