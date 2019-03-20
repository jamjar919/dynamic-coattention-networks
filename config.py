class Config:
    def __init__(self):
        self.BATCH_SIZE = 16
        self.EMBEDDING_DIMENSION = 300
        self.MAX_EPOCHS = 10
        self.HIDDEN_UNIT_SIZE = 200
        self.POOL_SIZE = 16

        self.QUESTION_FILE = 'data/dev.json'
        self.EMBEDDING_FILE = 'data/glove.6B.300d.txt'

        self.MAX_CONTEXT_LENGTH = 420
        self.MAX_QUESTION_LENGTH = 40
CONFIG = Config();