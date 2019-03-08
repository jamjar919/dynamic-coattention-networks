import tensorflow as tf

def transpose(tensor):
    return tf.transpose(tensor,perm=[0,2,1])

def to3D(matrix):
    return tf.reshape(matrix,[matrix.shape[0],matrix.shape[1],1])

def highway_network(U, lstm_hidden_state,
                    coattention_encoding_of_prev_start_word,
                    coattention_encoding_of_prev_end_word, hidden_unit_size = 200, pool_size = 16):
    U_transpose = tf.transpose(U,perm=[2, 1, 0]) 
    print("U_transpose.shape: ", U_transpose.shape)
    
    fn = lambda batch_of_word_encodings : highway_network_batch(tf.transpose(batch_of_word_encodings,perm = [1,0]), lstm_hidden_state,
                    coattention_encoding_of_prev_start_word,
                    coattention_encoding_of_prev_end_word) # Pass 1 batch consisting of 1 of the 632 words to the HMN. 
    
    result = tf.map_fn(fn, U_transpose) 
    print ("result.shape: ", result.shape) # result is now shape 632 * B * 1  where B is batch size. We have α_1 to α_632
    index = tf.argmax(result, axis=0, output_type=tf.int32) # Get argmax of α_1 to α_632 for each batch element. 
    print ("index.shape: ", index.shape)
    # Remove extra array wrap at the end
    result = tf.reshape(result, [result.shape[0], result.shape[1]]) # This is α_1,...α_m in equation 6 (or beta in equation 7)
    print("result after reshaping.shape ", result.shape)  # Make it 632xB instaed of 632xBx1
    return index, result

def highway_network_batch(batch_of_word_encodings,
                    lstm_hidden_state,
                    coattention_encoding_of_prev_start_word,
                    coattention_encoding_of_prev_end_word, hidden_unit_size = 200, pool_size = 16):

    # Get the scoped variables if they exist (they should)
    weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    wd = tf.get_variable("wd", shape=[hidden_unit_size, 5 * hidden_unit_size],
                                    initializer=weight_initer)
    w1 = tf.get_variable("w1", shape=[pool_size, hidden_unit_size, 3 * hidden_unit_size],
                                    initializer=weight_initer)
    w2 = tf.get_variable("w2", shape=[pool_size, hidden_unit_size, hidden_unit_size],
                                    initializer=weight_initer)
    w3 = tf.get_variable("w3", shape=[pool_size, 1, 2 * hidden_unit_size],
                                    initializer=weight_initer)
    b1 = tf.get_variable("b1", shape=[pool_size, hidden_unit_size])
    b2 = tf.get_variable("b2", shape=[pool_size, hidden_unit_size])
    b3 = tf.get_variable("b3", shape=[pool_size])

    batch_of_word_encodings = to3D(batch_of_word_encodings)
    print("batch_of_word_encodings shape: ",batch_of_word_encodings.shape)
    
    #From equation 10, concatenate h_i with u_{s_i - 1} with u_{e_i - 1}
    con = tf.concat(values=[lstm_hidden_state, coattention_encoding_of_prev_start_word,
                               coattention_encoding_of_prev_end_word], axis=1)
    con = to3D(con)
    print("con.shape:", con.shape)
    print("wd.shape: ", wd.shape)
    linear_model = tf.map_fn(lambda x: tf.matmul(wd,x),con) 
    print("linear_model shape: ",linear_model.shape)
    activated_value = tf.nn.tanh(linear_model) # This is "r" from eq.10
    print("activated_value shape: ",activated_value.shape)
    # For eq.11 concatenate u_t with r.
    con2 = tf.concat(values=[batch_of_word_encodings, activated_value],axis = 1) 
    print("con2.shape: ", con2.shape)

    # Calculate mt1. Multiplying W1 of shape p*l*3l with con2 of shape B*3l*1
    mt1_premax = tf.map_fn(lambda x: tf.map_fn(lambda wmat: tf.matmul(wmat,x), w1 ), con2)
    print("mt1_premax.shape: ",mt1_premax.shape)
    b1 = to3D(b1)
    mt1_premax = tf.map_fn(lambda x: x+b1, mt1_premax) # Add on the biases. 
    mt1_postmax =  tf.reduce_max(mt1_premax, axis=1) 
    print("mt1_postmax.shape : ",mt1_postmax.shape) # Of shape B*200*1

    # Calculate mt2.    W2 of dim p*l*l and mt1 of dim B*l*1
    mt2_premax = tf.map_fn(lambda x: tf.map_fn(lambda wmat: tf.matmul(wmat, x), w2), mt1_postmax)
    print("mt2_premax.shape", mt2_premax.shape)
    b2 = to3D(b2)
    mt2_premax = tf.map_fn(lambda x: x+b2, mt2_premax)
    mt2_postmax = tf.reduce_max(mt2_premax, axis=1)
    print("mt2_postmax.shape : ", mt2_postmax.shape) # This is 10*200*1

    #calculate the final HMN output. Equation 9
    mt1mt2 = tf.concat(values=[mt1_postmax, mt2_postmax], axis=1) 
    print("mt1mt2.size:", mt1mt2.shape)
    hmn_premax = tf.map_fn(lambda x: tf.map_fn(lambda wmat: tf.matmul(wmat, x), w3), mt1mt2) #
    hmn_premax = tf.reshape(hmn_premax, [hmn_premax.shape[0], hmn_premax.shape[1]])
    print("hmn_premax",hmn_premax.shape) #  Shape Bx16
    hmn_premax = tf.map_fn(lambda x: x+b3, hmn_premax)
    hmn_premax = to3D(hmn_premax) 
    hmn_postmax = tf.reduce_max(hmn_premax, axis=1)
    print("hmn shape: ", hmn_postmax.shape) # Shape Bx1. One argmax for each batch element. 

    # the hmn_postmax shape is 10. this is for each word in the doc. Do that for 632 words and take max
    return hmn_postmax