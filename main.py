import tensorflow as tf

# This file runs the actual question answering program using our trained network
saver = tf.train.import_meta_graph('model/saved.meta')
with tf.Session() as sess:
    # To initialize values with saved data
    saver.restore(sess, 'model/saved.data-1000-00000-of-00001')
    print(sess.run(global_step_tensor)) # returns 1000
