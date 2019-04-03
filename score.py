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
        for i in range(predicted_labels.shape[0]):
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
    
    def update_stats(self) :
        self.precision = self.true_positives / (self.true_positives + self.false_positives)
        self.recall = self.true_positives / (self.true_positives+self.false_negatives)
        self.F1 = 2 * (self.precision * self.recall) / (self.precision * self.recall)
        self.accuracy = (self.true_positives + self.true_negatives) / (self.true_positives + 
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
