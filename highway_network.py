import tensorflow as tf

def highway_network(kr,hs,us,ue,hidden_unit_size = 200, pool_size = 16):
    
    #print "kr",kr.shape
    #calculate r
    wd = tf.get_variable(name="wd",shape=[hidden_unit_size,5*hidden_unit_size],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    x = tf.concat([hs,us,ue],axis=1)
    r = tf.nn.tanh(tf.matmul(x,tf.transpose(wd)))
    #print r.shape

    #calculate mt1
    r1 = tf.expand_dims(tf.ones([int(kr.shape[1]),1]), 1) * r
    r1 = tf.reshape(r1,[-1,hidden_unit_size])
    kr1 = tf.reshape(kr,[-1,2*hidden_unit_size])
    krr1 = tf.concat([kr1,r1],axis=1)
    w1 = tf.get_variable(name="w1",shape=[pool_size*hidden_unit_size,3*hidden_unit_size],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b1 = tf.Variable(tf.constant(0.0,shape=[pool_size*hidden_unit_size,]),dtype=tf.float32)
    x1 = tf.matmul(krr1,tf.transpose(w1))+b1
    x1 = tf.reshape(x1,[-1,pool_size])
    x1 = tf.reduce_max(x1,axis=1)
    m1 = tf.reshape(x1,[-1,hidden_unit_size])
    #print m1.shape
    
    #calculate mt2
    w2 = tf.get_variable(name="w2",shape=[pool_size*hidden_unit_size,hidden_unit_size],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b2 = tf.Variable(tf.constant(0.0,shape=[pool_size*hidden_unit_size,]),dtype=tf.float32)
    x2 = tf.matmul(m1,tf.transpose(w2))+b2
    x2 = tf.reshape(x2,[-1,pool_size])
    x2 = tf.reduce_max(x2,axis=1)
    m2 = tf.reshape(x2,[-1,hidden_unit_size])
    #print m2.shape
    
    #max
    m1m2 = tf.concat([m1,m2],axis=1)
    #print "m1m2",m1m2.shape
    w3 = tf.get_variable(name="w3",shape=[pool_size,2*hidden_unit_size],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b3 = tf.Variable(tf.constant(0.0,shape=[pool_size,]),dtype=tf.float32)
    x3 = tf.matmul(m1m2,tf.transpose(w3))+b3
    #print x3.shape
    x3 = tf.reduce_max(x3,axis=1)
    #print x3.shape
    x3 = tf.reshape(x3,[-1,int(kr.shape[1])])
    #print "x3",x3.shape
    #argmax
    output = tf.argmax(x3,axis=1)
    output = tf.cast(output,dtype=tf.int32)
    
    return output,x3
