# This file trains the neural network using the encoder and decoder.

import numpy as np
import tensorflow as tf
import pandas as pd

i = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(i)
