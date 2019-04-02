import tensorflow as tf
import numpy as np
from network.config import CONFIG

# # Returns two masks. One that will help us get the argmax (ninf_mask) and other to mask logits for the loss function (one_zero_mask)
def getMask(seq_length, max_seq_length, val_one, val_two):
    mask =  tf.map_fn(lambda x: tf.concat([val_one * tf.ones([1, x], dtype=tf.float32), val_two * tf.ones([1, max_seq_length - x], dtype = tf.float32)], axis = 1), seq_length, dtype = tf.float32)
    return tf.squeeze(mask)

def highway_network(U, hs, u_s, u_e, context_seq_length, hidden_unit_size , pool_size):
    keep_rate = 1

    ''' Get the weights and biases for the network '''
    wd = tf.get_variable(name="wd",shape=[hidden_unit_size, 5*hidden_unit_size], dtype=tf.float32)
    w1 = tf.get_variable(name="w1",shape=[pool_size, hidden_unit_size, 3 * hidden_unit_size], dtype=tf.float32)
    b1 = tf.get_variable(name="b1" ,shape=[pool_size, hidden_unit_size,],dtype=tf.float32)
    w2 = tf.get_variable(name="w2",shape=[pool_size, hidden_unit_size, hidden_unit_size], dtype=tf.float32)
    b2 = tf.get_variable(name="b2" ,shape=[pool_size, hidden_unit_size,],dtype=tf.float32)
    w3 = tf.get_variable(name="w3",shape=[pool_size, 1, 2*hidden_unit_size], dtype=tf.float32)
    b3 = tf.get_variable(name="b3" ,shape=[pool_size,1],dtype=tf.float32)
    
    ''' Calculate r from equation 10 ''' 
    x = tf.concat([hs,u_s,u_e],axis=1)
    print("hs.shape :", hs.shape)
    print("us.shape: ", u_s.shape)
    print("ue.shape: ",u_e.shape)
    r = tf.nn.tanh(tf.matmul(x,tf.transpose(wd))) # Product of this is 10x200 (10x1000 * 1000x200)
    print("r.shape: ", r.shape)

    ''' Calculate mt1 (equation 11)   '''     
    r1 = tf.stack([r] * U.shape[1])   # Make 632 copies of r to get 632x10x200. 
    r1 = tf.transpose(r1, perm = [1,0,2]) #  Transpose to 10x632x200
    print("r1.shape at line 216 ", r1.shape)
    print("U.shape: ", U.shape)
    U_r1_concat = tf.concat([U,r1],axis=2) # Concat 10x632x200 and 10x632x400 to get 10x632x600
    U_r1_concat_dropout = tf.nn.dropout(U_r1_concat, keep_prob = keep_rate)
    print("U_r1_concat.shape at line 220 ", U_r1_concat.shape)
    x1 = tf.tensordot(U_r1_concat_dropout, w1, axes = [[2], [2]])  + b1 # make u_r1_concat_dropout for dropout. 
    print("x1.shape at line 242: ", x1.shape)
    m1 = tf.reduce_max(x1,axis=2)
    print("m1.shape: ", m1.shape)
    
    ''' Calculate mt2 (equation 12) '''
    m1_dropout = tf.nn.dropout(m1, keep_prob = keep_rate)
    m2_premax = tf.tensordot(m1_dropout, w2, axes = [[2], [2]]) + b2 # make m1_dropout for dropout. 
    print("m2_premax.shape: ", m2_premax.shape)
    m2 = tf.reduce_max(m2_premax, axis = 2)
    print("m2.shape: ", m2.shape)
    
    # Calculate HMN max.
    m1m2 = tf.concat([m1,m2],axis=2)
    m1m2 = tf.nn.dropout(m1m2, keep_prob = keep_rate)
    print ("m1m2.shape: ",m1m2.shape)
    x3 = tf.tensordot(m1m2, w3, axes = [[2], [2]]) + b3
    print("x3.shape: ", x3.shape)
    x3 = tf.squeeze(tf.reduce_max(x3,axis=2)) # Remove dimension of size 1
    #x3 = tf.Print(x3, [x3[0][0:2]], "x3 (0:2) before mask")
    #x3 = tf.Print(x3, [x3[0][600:602]], "x3 (600:602) before mask")

    print ("x3.shape: ", x3.shape)
    ninf_mask = getMask(context_seq_length, CONFIG.MAX_CONTEXT_LENGTH, val_one = 0., val_two = -10**30) # Get two masks from the sequence length (calculated in encoder)
    print("ninf mask shape: ", ninf_mask.shape)
    x3_ninf_mask = x3 + ninf_mask # Ignore elements which were simply padded on. (element wise multiplication)
    #x3_ninf_mask = tf.Print(x3_ninf_mask, [x3_ninf_mask[0][0:2]], "x3 (0:2) after ninf mask") # Check that the start words are unaffected
    #x3_ninf_mask = tf.Print(x3_ninf_mask, [x3_ninf_mask[0][600:602]], "x3 (600:602) after ninf mask") # Check that the probably padded words are affected.
        
    output = tf.argmax(x3_ninf_mask, axis=1) # Take argmax from the mask.
    print("1st output shape: ", output.shape)
    output = tf.squeeze(tf.cast(output, dtype=tf.int32)) # Remove dimensions of size 1
    print("2nd output shape: ", output.shape)

    return output, x3_ninf_mask