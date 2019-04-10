import __init__
import tensorflow as tf
import numpy as np
from network.config import CONFIG

# Returns 1D mask for every batch element in order to mask out the HMN output scores. 
def getMask(seq_length, max_seq_length, val_one, val_two):
    mask =  tf.map_fn(lambda x: tf.concat([val_one * tf.ones([1, x], dtype=tf.float32), val_two * tf.ones([1, max_seq_length - x], dtype = tf.float32)], axis = 1), seq_length, dtype = tf.float32)
    return tf.squeeze(mask)

def highway_network(U, hs, u_s, u_e, context_seq_length, max_context_length, dropout_keep_rate, hidden_unit_size , pool_size):

    ''' Get the weights and biases for the network '''
    Wd = tf.get_variable(name="Wd",shape=[hidden_unit_size, 5*hidden_unit_size], dtype=tf.float32)
    W1 = tf.get_variable(name="W1",shape=[pool_size, hidden_unit_size, 3 * hidden_unit_size], dtype=tf.float32)
    b1 = tf.get_variable(name="b1" ,shape=[pool_size, hidden_unit_size,],dtype=tf.float32)
    W2 = tf.get_variable(name="W2",shape=[pool_size, hidden_unit_size, hidden_unit_size], dtype=tf.float32)
    b2 = tf.get_variable(name="b2" ,shape=[pool_size, hidden_unit_size,],dtype=tf.float32)
    W3 = tf.get_variable(name="W3",shape=[pool_size, 1, 2*hidden_unit_size], dtype=tf.float32)
    b3 = tf.get_variable(name="b3" ,shape=[pool_size,1],dtype=tf.float32)
    
    ''' Calculate r from equation 10 ''' 
    r = tf.concat([hs,u_s,u_e],axis=1)
    print("hs.shape :", hs.shape)
    print("us.shape: ", u_s.shape)
    print("ue.shape: ",u_e.shape)
    r = tf.nn.tanh(tf.matmul(r,tf.transpose(Wd))) # Product of this is 10x200 (10x1000 * 1000x200)
    print("r.shape: ", r.shape)

    ''' Calculate m1 (equation 11)   '''     
    r_stacked = tf.stack([r] * U.shape[1])   # Make r to get max_context_length x batch_size x hidden_state_size. 
    r_stacked = tf.transpose(r_stacked, perm = [1,0,2]) #  Transpose to batch_size x max_context_length x hidden_state_size
    print("r1.shape at line 216 ", r_stacked.shape)
    print("U.shape: ", U.shape)
    U_r1_concat = tf.concat([U,r_stacked],axis=2) # batch_size x max_context_length x 2*hidden_state_size
    U_r1_concat = tf.nn.dropout(U_r1_concat,  keep_prob= dropout_keep_rate )
    # (batch_size x max_context_length x 2*hidden_state_size) , (pool_size x hidden_unit_size x 3 * hidden_unit_size)
    print("U_r1_concat.shape at line 220 ", U_r1_concat.shape)
    m1 = tf.tensordot(U_r1_concat, W1, axes = [[2], [2]])  + b1  # batch_size x max_context_length x pool_size x hidden_unit_size
    print("x1.shape at line 242: ", m1.shape) 
    m1 = tf.reduce_max(m1,axis=2) #  batch_size x max_context_length x hidden_unit_size
    print("m1.shape: ", m1.shape)
    
    ''' Calculate m2 (equation 12) '''
    # (batch_size x max_contxt_length x hidden_unit_size) , (pool_size, hidden_unit_size, hidden_unit_size)
    m2 = tf.tensordot(m1, W2, axes = [[2], [2]]) + b2  # batch_size x max_context_length x pool_size x hidden_unit_size
    m2 = tf.nn.dropout(m2,  keep_prob= dropout_keep_rate )
    print("m2.shape: ", m2.shape)
    m2 = tf.reduce_max(m2, axis = 2) # batch_size x max_context_length x hidden_unit_size
    print("m2.shape: ", m2.shape)
    
    # Calculate HMN max.
    m1m2 = tf.concat([m1,m2],axis=2) # batch_size x max_context_length x 2*hidden_unit_size
    print ("m1m2.shape: ",m1m2.shape)
    # (batch_size x max_context_length x 2*hidden_unit_size) , (pool_size, 1, 2*hidden_unit_size)
    m1m2_concat = tf.tensordot(m1m2, W3, axes = [[2], [2]]) + b3  # batch_size x max_context_length x 1 x 2*hidden_unit_size
    m1m2_concat = tf.nn.dropout(m1m2_concat,  keep_prob= dropout_keep_rate )
    print("m1m2_concat.shape: ", m1m2_concat.shape)
    logits = tf.squeeze(tf.reduce_max(m1m2_concat,axis=2)) # batch_size x max_context_length 
    
    print ("logits.shape: ", logits.shape)
    ninf_mask = getMask(context_seq_length, max_context_length, val_one = 0., val_two = -10**30) # Get two masks from the sequence length (calculated in encoder)
    print("ninf mask shape: ", ninf_mask.shape)
    masked_logits = logits + ninf_mask # Ignore elements which were simply padded on. (element wise multiplication)

    output = tf.argmax(masked_logits, axis=1) # Take argmax from the mask.
    print("1st output shape: ", output.shape)
    output = tf.squeeze(tf.cast(output, dtype=tf.int32)) # Remove dimensions of size 1
    print("2nd output shape: ", output.shape)

    return output, masked_logits