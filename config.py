class Config:
    def __init__(self):
        self.BATCH_SIZE = 64
        self.EMBEDDING_DIMENSION = 300
        self.MAX_EPOCHS = 100
        self.HIDDEN_UNIT_SIZE = 200
        self.POOL_SIZE = 4

        self.LEARNING_RATE = 0.001
        self.CLIP_NORM = 3.0
        self.DROPOUT_KEEP_PROB = 0.7

        self.QUESTION_FILE = 'data/train.json'
        self.EMBEDDING_FILE = 'data/glove.840B.300d.txt'

        self.MAX_CONTEXT_LENGTH = 632
        self.MAX_QUESTION_LENGTH = 33
CONFIG = Config();