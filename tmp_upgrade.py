def recurrent_neural_network(x):
    #####################################################################

    W = {
        'hidden': tf.Variable(tf.random.normal([chunk_size, rnn_size])),
        'output': tf.Variable(tf.random.normal([rnn_size, n_classes]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([rnn_size], mean=1.0)),
        'output': tf.Variable(tf.random_normal([n_classes]))
    }

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.nn.relu(tf.matmul(x, W['hidden']) + biases['hidden'])
    x = tf.split(x, n_chunks, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    Dropout = tf.contrib.rnn.DropoutWrapper(lstm_cell_1, output_keep_prob=0.5)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2, Dropout], state_is_tuple=True)
    # Get LSTM cell output
    outputs, final_states = tf.contrib.rnn.static_rnn(lstm_cells, x, dtype=tf.float32)
    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    #    lstm_last_output=tf.transpose(outputs, [1,0,2])
    # Linear activation

    return tf.matmul(outputs[-1], W['output']) + biases['output']
