# Question Answering using Dynamic Coattention Networks

https://arxiv.org/pdf/1611.01604.pdf

## Running the program
### Config
The file located at `network/config.py` supplies the config for most aspects of the model. A sample is attached below.

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
            self.BILSTM_DROPOUT_KEEP_PROB = 1
            self.TRAIN_PERCENTAGE = 0.90

            self.QUESTION_FILE = 'data/train.json'
            self.EMBEDDING_FILE = 'data/glove.840B.300d.txt'

            self.MAX_CONTEXT_LENGTH = 632
            self.MAX_QUESTION_LENGTH = 40

    CONFIG = Config()

### Training
Run `train.py` to train the model. This loads cached word embeddings from the /data/ folder, or generates and saves them if they haven't been generated before. You can supply additional parameters:
 - `--regenerateEmbeddings` to force generating and resaving the cached word embeddings to disk 
 - `--noGPU` to disable running with the GPU.
 
The training will automatically test on a validation subset, and log loss, f1, and exact match statistics to CSV in the /results/ folder. The model is also saved every epoch in the /models/ folder.

### Evaluating
To evaluate F1/EM performance on the dev set, the file `test_saved_model.py` is what you need. You can also use the file `main.py` to see the example output of the network on a random context and question. You can also use `testing.py` to ask a question and obtain a context to search with from wikipedia.

## Results
Our standard model achieved a maximum F1 of 71.8% and EM of ?  on the dev set (which we used as our test set) after 7 epochs, compared to the paper’s F1 of 75.9%

![](https://raw.githubusercontent.com/jamjar919/dynamic-coattention-networks/master/loss_graph.png?token=AFYtbLdhVAmZilLdFlcjc5oMH0_SQslGks5crPkfwA%3D%3D)

![](https://raw.githubusercontent.com/jamjar919/dynamic-coattention-networks/master/loss_validation_loss_graph.png?token=AFYtbDFLaIc-BqPfHpVwKztj5_yKBoQtks5crPlTwA%3D%3D)

### Dataset Statistics

![](https://github.com/jamjar919/dynamic-coattention-networks/blob/master/data/dev_question_histogram.png?raw=true)
![](https://github.com/jamjar919/dynamic-coattention-networks/blob/master/data/train_question_histogram.png?raw=true)

### Example Context Heatmap (zoom in)

![](https://github.com/jamjar919/dynamic-coattention-networks/blob/master/question.png?raw=true)
