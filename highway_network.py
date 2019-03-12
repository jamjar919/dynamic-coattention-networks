import tensorflow as tf

def highway_network(U, hs, u_s, u_e, hidden_unit_size , pool_size):
    
    ''' Get the weights and biases for the network '''
    wd = tf.get_variable(name="wd",shape=[hidden_unit_size, 5*hidden_unit_size], dtype=tf.float32)
    w1 = tf.get_variable(name="w1",shape=[pool_size, hidden_unit_size, 3*hidden_unit_size], dtype=tf.float32)
    b1 = tf.Variable(tf.constant(0.0,shape=[pool_size, hidden_unit_size,]),dtype=tf.float32)
    w1_T = tf.transpose(w1, perm = [2,0,1]) # Conver to 600x16x200
    w2 = tf.get_variable(name="w2",shape=[pool_size, hidden_unit_size, hidden_unit_size], dtype=tf.float32)
    b2 = tf.Variable(tf.constant(0.0,shape=[pool_size, hidden_unit_size, ]),dtype=tf.float32)
    w2_T = tf.transpose(w2, perm = [2,0,1])
    w3 = tf.get_variable(name="w3",shape=[pool_size, 1, 2*hidden_unit_size], dtype=tf.float32)
    w3_T = tf.transpose(w3, perm = [2,0,1])
    b3 = tf.Variable(tf.constant(0.0,shape=[pool_size, 1]), dtype=tf.float32)
    
    # Calculate r 
    x = tf.concat([hs,u_s,u_e],axis=1)
    print("hs.shape :", hs.shape)
    print("us.shape: ", u_s.shape)
    print("ue.shape: ",u_e.shape)
    r = tf.nn.tanh(tf.matmul(x,tf.transpose(wd))) # This is 10x200
    print("r.shape: ", r.shape)

    #calculate mt1    # TENSORDOT (A,B, axes = [])
    r1 = tf.stack([r] * U.shape[1])   # 632x10x200, to accommodate max context length. 
    r1 = tf.transpose(r1, perm = [1,0,2]) #  Transpose to 10x632x200
    print("r1.shape at line 216 ", r1.shape)
    print("U.shape: ", U.shape)
    U_r1_concat = tf.concat([U,r1],axis=2) # Concat 10x632x200 and 10x632x400 to get 10x632x600
    print("U_r1_concat.shape at line 220 ", U_r1_concat.shape)
    
    print("w1_T.shape: ", w1_T.shape)
    x1 = tf.tensordot(U_r1_concat, w1_T, axes = [[2], [0]])  + b1
    print("x1.shape at line 242: ", x1.shape)
    m1 = tf.reduce_max(x1,axis=2)
    print("m1.shape: ", m1.shape)
    
    #calculate mt2
    
    print ("w2_t.shape: ", w2_T.shape)
    m2_premax = tf.tensordot(m1, w2_T, axes = [[2], [0]]) + b2
    print("m2_premax.shape: ", m2_premax.shape)
    m2 = tf.reduce_max(m2_premax, axis = 2)
    print("m2.shape: ", m2.shape)
    
    # Calculate HMN max.
    m1m2 = tf.concat([m1,m2],axis=2)
    print ("m1m2.shape: ",m1m2.shape)
    print("w3_T.shape: ", w3_T.shape)
    x3 = tf.tensordot(m1m2,w3_T, axes = [[2], [0]]) + b3
    print("x3.shape: ", x3.shape)
    x3 = tf.squeeze(tf.reduce_max(x3,axis=2)) # Remove dimension of size 1
    print ("x3.shape: ", x3.shape)
    output = tf.argmax(x3,axis=1)
    print("1st output shape: ", output.shape)
    output = tf.squeeze(tf.cast(output,dtype=tf.int32)) # Remove dimensions of size 1
    print("2nd output shape: ", output.shape)

    return output,x3