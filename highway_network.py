import tensorflow as tf

def highway_network(U, hs, u_s, u_e, hidden_unit_size , pool_size):
    dropout_rate = 0.5

    ''' Get the weights and biases for the network '''
    wd = tf.get_variable(name="wd",shape=[hidden_unit_size, 5*hidden_unit_size], dtype=tf.float32)
    w1 = tf.get_variable(name="w1",shape=[pool_size, hidden_unit_size, 3 * hidden_unit_size], dtype=tf.float32)
    b1 = tf.Variable(tf.constant(0.0,shape=[pool_size, hidden_unit_size,]),dtype=tf.float32)
    w2 = tf.get_variable(name="w2",shape=[pool_size, hidden_unit_size, hidden_unit_size], dtype=tf.float32)
    b2 = tf.Variable(tf.constant(0.0,shape=[pool_size, hidden_unit_size, ]),dtype=tf.float32)
    w3 = tf.get_variable(name="w3",shape=[pool_size, 1, 2*hidden_unit_size], dtype=tf.float32)
    b3 = tf.Variable(tf.constant(0.0,shape=[pool_size, 1]), dtype=tf.float32)
    
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
    U_r1_concat_dropout = tf.nn.dropout(U_r1_concat, keep_prob = dropout_rate)
    print("U_r1_concat.shape at line 220 ", U_r1_concat.shape)
    # w1 is of shape 16*200*600 UU_r1_concat_dropout is 10x632x600 b1 is 16*200
    x1 = tf.tensordot(U_r1_concat_dropout, w1, axes = [[2], [2]])  + b1
    print("x1.shape at line 242: ", x1.shape) #10 632 16 200
    m1 = tf.reduce_max(x1,axis=2) #10*632*200
    print("m1.shape: ", m1.shape)
    
    ''' Calculate mt2 (equation 12) '''
    m1_dropout = tf.nn.dropout(m1, keep_prob = dropout_rate)
    #w is 16*200*200 m1_dropout 10*632*200
    m2_premax = tf.tensordot(m1_dropout, w2, axes = [[2], [2]]) + b2
    print("m2_premax.shape: ", m2_premax.shape)
    m2 = tf.reduce_max(m2_premax, axis = 2)
    print("m2.shape: ", m2.shape)
    
    # Calculate HMN max.
    m1m2 = tf.concat([m1,m2],axis=2)
    m1m2 = tf.nn.dropout(m1m2, keep_prob = dropout_rate)
    print ("m1m2.shape: ",m1m2.shape) #10*632*400
    x3 = tf.tensordot(m1m2, w3, axes = [[2], [2]]) + b3
    print("x3.shape: ", x3.shape) # 10*632*16*1
    x3 = tf.squeeze(tf.reduce_max(x3,axis=2)) # Remove dimension of size 1
    #x3 contains the alpha values for each of the 632 words in the context question for all the contexts in the batch
    print ("x3.shape: ", x3.shape)
    output = tf.argmax(x3,axis=1)
    print("1st output shape: ", output.shape)
    output = tf.squeeze(tf.cast(output,dtype=tf.int32)) # Remove dimensions of size 1
    print("2nd output shape: ", output.shape)

    # return the arg of the word that is to be considered as the start/end word and x3
    return output,x3