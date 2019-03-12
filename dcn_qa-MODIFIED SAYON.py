
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


# In[2]:


def preprocess(filepath,ml):
    c = []
    if ml == 0:
        ml = 1000
    with open(filepath,'r') as f:
        s = 0
        for line in f:
            line = line.replace('\n','')
            s = s+1
            c.append([int(line.split()[i]) for i in range(len(line.split())) if i<ml])
            if s>200:
                break
    return c

def final_preprocess(c):   #padding
    max_l = max([len(i) for i in c])
    for i in range(len(c)):
        c[i] = c[i] + (max_l-len(c[i]))*[0]
    return c,max_l


# In[9]:
#dummy data section
#size=50
#context_data = context_data[:size]
#question_data = question_data[:size]
#answer_start_data = answer_start_data[:size]
#answer_end_data = answer_end_data[:size]


#reading train data and preprocessing them + padding
#pad the context and question data
context_data,max_l_context = final_preprocess(preprocess('data/train.ids.context',600))
question_data,max_l_question = final_preprocess(preprocess('data/train.ids.question',0))
answer_data = preprocess('data/train.span',0)
answer_start_data = [i[0] for i in answer_data]
answer_end_data = [i[1] for i in answer_data]

print(len(context_data))

# In[10]:


print (max_l_context)
print (max_l_question)


# In[11]:


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
    length = tf.reduce_sum(used, axis=1)
    length = tf.cast(length, tf.int32)
    #print "sequence length tf shape:",length.shape
    return length


# In[36]:


#encoder function
def encoder(question,context,embeddings,hidden_units=200):
    batch_size = 10 #tf.shape(question)[0]
    #question and document encoder
    q_embedding = tf.nn.embedding_lookup(embeddings,question)
    d_embedding = tf.nn.embedding_lookup(embeddings,context)
    
    print(q_embedding.shape,"=?,60,100")
    print(d_embedding.shape,"=?,600,100")
    
    lstm_enc = tf.contrib.rnn.BasicLSTMCell(hidden_units)
    
    with tf.variable_scope('document_encoder') as scope1:
        document_states,_ = tf.nn.dynamic_rnn(cell=lstm_enc,
                                              dtype=tf.float32,
                                              inputs=d_embedding,
                                              sequence_length=length(q_embedding),
                                              time_major=False)
        
    with tf.variable_scope('question_encoder') as scope2:
        question_states,_ = tf.nn.dynamic_rnn(cell=lstm_enc,
                                              dtype=tf.float32,
                                              inputs=q_embedding,
                                              sequence_length=length(d_embedding),
                                              time_major=False)

    Wq = tf.get_variable(name="Wq",shape=[hidden_units,hidden_units],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    bq = tf.Variable(tf.constant(0.0,shape=[hidden_units,]),dtype=tf.float32,name='bq')
    Wq = tf.expand_dims(tf.ones([batch_size,1]), 1) * Wq
    #question_states_new = tf.reshape(question_states,shape=[-1,hidden_units])
    print(document_states.shape,"=?,600,200")
    print(question_states.shape,"=?,60,200")
    #print question_states_new.shape,"=?,200"
    print(Wq.shape,"=?,200,200")
    print(bq.shape,"=200,")
    
                    
    question_states_modified_duplicate = tf.nn.tanh(tf.matmul(question_states,Wq)+bq)
    #question_states_modified_duplicate = tf.reshape(question_states_modified,shape=[-1,int(question_states.shape[1]),hidden_units])
    question_states_modified = tf.transpose(question_states_modified_duplicate,perm=[0,2,1]) #tf.reshape(question_states_modified,shape=[-1,hidden_units,int(question_states.shape[1])])
    print(question_states_modified.shape,"=?,200,60")
    print(question_states_modified_duplicate.shape,"=?,60,200")

    #coattention encoder
    
    l = tf.matmul(document_states,question_states_modified)
    print(l.shape,"=?,600,60")
    aq = tf.nn.softmax(l)
    ad = tf.nn.softmax(tf.transpose(l,perm=[0, 2, 1]))
    print(aq.shape,"=?,600,60")
    print(ad.shape,"=?,60,600")
    
    cq = tf.matmul(tf.transpose(aq,perm=[0,2,1]),document_states)
    print(cq.shape,"=?,60,200")
    print(question_states_modified_duplicate.shape,"=?,60,200")
    qcq = tf.concat([question_states_modified_duplicate,cq],2)
    print(qcq.shape,"=?,60,400")
    cd = tf.matmul(tf.transpose(ad,perm=[0,2,1]),qcq)
    print(cd.shape,"=?,600,400")
    dcd = tf.concat([document_states,cd],axis=2)
    
    with tf.variable_scope('coattention'):
        u_lstm_fw = tf.contrib.rnn.BasicLSTMCell(hidden_units)  #bi-lstm
        u_lstm_bw = tf.contrib.rnn.BasicLSTMCell(hidden_units)
        u_states,_ = tf.nn.bidirectional_dynamic_rnn(cell_bw=u_lstm_bw,cell_fw=u_lstm_fw,dtype=tf.float32,inputs=dcd,time_major=False,sequence_length=length(dcd))
    encoder_states = tf.concat(u_states,2)
    print("encoder states. shape: ", encoder_states.shape)
    return encoder_states


# In[61]:


#decoder function
def decoder(knowledge_reps,hidden_units = 200):
    #randomly initialise s and e
    batch_size = 10#tf.shape(knowledge_reps)[0]
    #print batch_size
    pool = 16
    e = np.random.randint(max_l_context) + 1
    s = np.random.randint(e)
    print("s: ", s)
    sv = tf.tile([s],[batch_size])
    ev = tf.tile([e],[batch_size])
    print("ev.shape ", ev.shape)
    print("ev: ",ev)

    #lstm cell
    #with tf.variable_scope('lstm_dec') as scope_dec:
    lstm_dec = tf.contrib.rnn.LSTMCell(hidden_units)
    ch = lstm_dec.zero_state(batch_size,dtype=tf.float32)
    hi,ci = ch
    
    
    with tf.variable_scope('hmn1') as scope1:
        wd = tf.get_variable(name="wd",shape=[hidden_units,5*hidden_units],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
        w1 = tf.get_variable(name="w1",shape=[pool,hidden_units,3*hidden_units],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
        w2 = tf.get_variable(name="w2",shape=[pool,hidden_units,hidden_units],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
        w3 = tf.get_variable(name="w3",shape=[pool,1,2*hidden_units],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    
    with tf.variable_scope('hmn2') as scope2:
        wd = tf.get_variable(name="wd",shape=[hidden_units,5*hidden_units],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
        w1 = tf.get_variable(name="w1",shape=[pool,hidden_units,3*hidden_units],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
        w2 = tf.get_variable(name="w2",shape=[pool,hidden_units,hidden_units],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
        w3 = tf.get_variable(name="w3",shape=[pool,1,2*hidden_units],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)

        
    #loop 4 times to call lstm cell to:
    for i in range(4):
        #concatenate u_s and u_e
        u_s = tf.gather_nd(params=knowledge_reps,indices=tf.stack([tf.range(batch_size,dtype=tf.int32),sv],axis=1))
        print("after gather us.shape ", u_s.shape)
        print("knowledge_reps.shape: ", knowledge_reps.shape)
        u_e = tf.gather_nd(params=knowledge_reps,indices=tf.stack([tf.range(batch_size,dtype=tf.int32),ev],axis=1))
        print("after gather ue.shape ", u_e.shape)
        usue = tf.concat([u_s,u_e],axis=1)
        print("usue.shape ", usue.shape)
        #print(i,usue.shape,hi.shape)
        #calculate hi
         
        with tf.variable_scope("hmn1",reuse=True) as scope1:
            sv,hmns_output = hmn(knowledge_reps,hi,u_s,u_e,hidden_units,pool)#loop over the document length times to obtain alpha t using HNM function
            print("sv.shape and hmns_output.shape: ", sv.shape, hmns_output.shape)
        with tf.variable_scope("hmn2",reuse=True) as scope2:
            ev,hmne_output = hmn(knowledge_reps,hi,u_s,u_e,hidden_units,pool)#loop over the document length times to obtain beta t using HNM function
        hmne_output = tf.Print(hmne_output, [hmne_output], "Inside loop")
    
        hi,ch = lstm_dec(inputs=usue,state=ch) 
        
    hmne_output = tf.Print(hmne_output, [hmne_output], "Outside loop")
    
    return sv,ev,hmns_output,hmne_output


# In[62]:


def hmn(kr,hs,us,ue,hidden_units,pool=16):
    
    #print "kr",kr.shape
    #calculate r
    wd = tf.get_variable(name="wd",shape=[hidden_units,5*hidden_units],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    x = tf.concat([hs,us,ue],axis=1)
    x = tf.Print(x,[x], "X IS BLA BLA BLA")
    print("hs.shape :", hs.shape)
    print("us.shape: ", us.shape)
    print("ue.shape: ",ue.shape)
    r = tf.nn.tanh(tf.matmul(x,tf.transpose(wd)))
    print("r.shape: ", r.shape)
    #print r.shape 

    #calculate mt1    # TENSORDOT (A,B, axes = [])
    ''' to be tested'''
    r1 = tf.stack([r]*kr.shape[1])   # 632x10x200
    r1 = tf.transpose(r1, perm = [1,0,2]) #  Transpose to 10x632x200
    print("r1.shape at line 214 ", r1.shape)
    #r1 = tf.reshape(r1,[-1,hidden_units])
    print("r1.shape at line 216 ", r1.shape)
    print("kr.shape: ", kr.shape)
    krr1 = tf.concat([kr,r1],axis=2) # Concat 10x632x200 and 10x632x400 to get 10x632x600
    print("krr1.shape at line 220 ", krr1.shape)
    w1 = tf.get_variable(name="w1",shape=[pool,hidden_units,3*hidden_units],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    print("w1.shape: ", w1.shape)
    b1 = tf.Variable(tf.constant(0.0,shape=[pool, hidden_units,]),dtype=tf.float32)
    w1_T = tf.transpose(w1, perm = [2,0,1]) # Conver to 600x16x200
    print("w1_T.shape: ", w1_T.shape)
    x1 = tf.tensordot(krr1, w1_T, axes = [[2], [0]])  + b1
    print("x1.shape at line 242: ", x1.shape)
    #x1 = tf.reshape(x1,[-1,pool])
    #print("x1.shape at line 244: ", x1.shape)
    m1 = tf.reduce_max(x1,axis=2)
    #print("x1.shape at line 246: ", x1.shape)
    #m1 = tf.reshape(x1,[-1,hidden_units])
    print("m1.shape: ", m1.shape)
    #print m1.shape
    
    #calculate mt2
    w2 = tf.get_variable(name="w2",shape=[pool,hidden_units,hidden_units],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b2 = tf.Variable(tf.constant(0.0,shape=[pool,hidden_units,]),dtype=tf.float32)
    #x2 = tf.matmul(m1,tf.transpose(w2))+b2
    w2_T = tf.transpose(w2, perm = [2,0,1])
    print ("w2_t.shape: ", w2_T.shape)
    m2_premax = tf.tensordot(m1, w2_T, axes = [[2], [0]])
    print("m2_premax.shape: ", m2_premax.shape)
    m2 = tf.reduce_max(m2_premax, axis = 2)
    print("m2.shape: ", m2.shape)
    #print m2.shape
    
    #max
    m1m2 = tf.concat([m1,m2],axis=2)
    print ("m1m2.shape: ",m1m2.shape)
    w3 = tf.get_variable(name="w3",shape=[pool,1,2*hidden_units],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float32)
    b3 = tf.Variable(tf.constant(0.0,shape=[pool,1]),dtype=tf.float32)
    w3_T = tf.transpose(w3, perm = [2,0,1])
    print("w3_T.shape: ", w3_T.shape)
    x3 = tf.tensordot(m1m2,w3_T, axes = [[2], [0]]) + b3
    print("x3.shape: ", x3.shape)
    #print x3.shape
    x3 = tf.reduce_max(x3,axis=2)
    print ("x3.shape: ", x3.shape)
    
    output = tf.argmax(x3,axis=1)
    print("1st output shape: ", output.shape)
    output = tf.cast(output,dtype=tf.int32)
    print("2nd output shape: ", output.shape)

    
    return output,x3


# In[63]:


#read embedding file
embedding_array = np.load('data/glove.trimmed.100.npz')


# In[64]:


## create placeholders
tf.reset_default_graph()
hidden_units = 200
question = tf.placeholder(dtype=tf.int32,shape=[None,max_l_question])
context = tf.placeholder(dtype=tf.int32,shape=[None,max_l_context])
answer_start = tf.placeholder(dtype=tf.int32,shape=[None])
answer_end = tf.placeholder(dtype=tf.int32,shape=[None])
embeddings = tf.constant(embedding_array['glove'],dtype=tf.float32)

encoder_states = encoder(question,context,embeddings)
print("Encoder states shape: ", encoder_states.shape)
print("decoder starts")
decoder_output_start, decoder_output_end, hmns_output, hmne_output = decoder(encoder_states)

## add loss
l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_start, logits=hmns_output)
l2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer_end, logits=hmne_output)
loss = l1 + l2
## add optimizer
train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)


# In[65]:


#initialise variables and train
init = tf.global_variables_initializer()


# In[ ]:


#session run train
with tf.Session() as sess:
    sess.run(init)
    batch_size = 10 # Was 50
    for epocs in range(1): # Was 200
        counter = 0
        for steps in range(4):
            question_batch = np.array(question_data[counter:(counter+batch_size)])
            context_batch = np.array(context_data[counter:(counter+batch_size)])
            answer_start_batch = np.array(answer_start_data[counter:(counter+batch_size)])
            answer_end_batch = np.array(answer_end_data[counter:(counter+batch_size)])
            sess.run(train_op,feed_dict = {question : question_batch, context : context_batch, answer_start : answer_start_batch, answer_end : answer_end_batch})
            if steps % 2 == 0:
                loss_val = sess.run(loss,feed_dict = {question : question_batch, context : context_batch, answer_start : answer_start_batch, answer_end : answer_end_batch})
                print ("epoch:",epocs,",","step:",steps,"loss:",np.mean(loss_val))
        counter = counter + batch_size
