import tensorflow as tf
import numpy

class SimpleLSTM:
    def __init__(self, sequence_length, max_len=50, hidden_layer=1280, num_classes=2):
        self.sequence_length = sequence_length
        self.hidden_layer = hidden_layer
        self.n_class = num_classes
        # Placeholders for input, output and dropout
        # self.x1_indexs = tf.placeholder(tf.int32, [None, sequence_length], name="x1_indexs")
        self.x1_inputs = tf.placeholder(tf.float32, [None, sequence_length, 46], name="x1_pos")

        # self.x2_indexs = tf.placeholder(tf.int32, [None, sequence_length], name="x2_indexs")
        self.x2_inputs = tf.placeholder(tf.float32, [None, sequence_length, 46], name="x2_pos")
        self.y = tf.placeholder(tf.float32, [None, num_classes], name="y")
        # self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # self.lr = tf.placeholder(tf.float32, name="learning_rate")
        # self.train_phase = tf.placeholder(tf.bool, name="phase_train")

        l2_loss = tf.constant(0.0)
        # Embedding layer
        # with tf.device('/cpu:0'), tf.name_scope("embedding"):
        #     normal_embedding_matrix = tf.nn.l2_normalize(embedding_matrix, dim=1)

        #     self.W = tf.Variable(normal_embedding_matrix, name="W_embedding")
        #     self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.x1_indexs)
        #     self.em_other1 = tf.concat(2, [self.embedded_chars, self.x1_pos])
        #     self.embedded_chars_expanded1 = tf.expand_dims(self.em_other1, -1)

        #     self.embedded_chars2 = tf.nn.embedding_lookup(self.W, self.x2_indexs)
        #     self.em_other2 = tf.concat(2, [self.embedded_chars, self.x2_pos])
        #     self.embedded_chars_expanded2 = tf.expand_dims(self.em_other2, -1)
        weights = {
                   'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_class]))
               }
        biases = {
                   'out': tf.Variable(tf.random_normal([self.n_class]))
               }
    # def RNN(x, weights, biases):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Permuting batch_size and n_steps
        x1 = tf.transpose(self.x1_inputs, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x1 = tf.reshape(x1, [-1, self.sequence_length])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x1 = tf.split(x1, self.sequence_length, 0)

        # Define a lstm cell with tensorflow
        lstm_cell1 = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs1, states1 = rnn.static_rnn(lstm_cell, x1, dtype=tf.float32)

        x2 = tf.transpose(self.x2_inputs, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x2 = tf.reshape(x2, [-1, self.sequence_length])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x2 = tf.split(x2, self.sequence_length, 0)

        # Define a lstm cell with tensorflow
        lstm_cell2 = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs2, states2 = rnn.static_rnn(lstm_cell, x2, dtype=tf.float32)


        W1 = tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden]))
        W2 = tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden]))

        o1 = tf.matmul(tf.matmul(outputs1[-1], W1), tf.transpose(outputs2[-1]))
        o2 = tf.matmul(tf.matmul(outputs2[-1], W2), tf.transpose(outputs2[-1]))
        # Linear activation, using rnn inner loop last output
        if (o1-o2>0.3):
            res = np.array([1, 0], dtype="float32")
        else:
            res = np.array([0, 1], dtype="float32")
        losses = tf.nn.softmax_cross_entropy_with_logits(res, self.y)
        self.loss = tf.reduce_mean(losses)
        self.optimizer = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(self.loss)