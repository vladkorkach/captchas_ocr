import tensorflow as tf
import network.utils as utils

FLAGS = utils.FLAGS
num_classes = utils.num_classes


class OCRNetwork:
    def __init__(self, mode):
        self.mode = mode
        # Placeholder for input data
        self.inputs = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
        # SparseTensor required by ctc_loss op
        self.labels = tf.sparse_placeholder(tf.int32)

        self._extra_train_ops = []

    def build_graph(self):
        self._build_model()
        self._build_train_op()
        # prepare var summary for tensorboard
        self.merged_summary = tf.summary.merge_all()

    def _build_model(self):
        cnn_params = [
            {"filters": FLAGS.image_channel, "k_size": 3, "use_pool": True, "padding": "SAME", "batch_norm": True},
            {"filters": 64, "k_size": 3, "use_pool": True, "padding": "SAME", "batch_norm": True},
            {"filters": 64, "k_size": 3, "use_pool": True, "padding": "SAME", "batch_norm": True},
            {"filters": 128, "k_size": 3, "use_pool": True, "padding": "SAME", "batch_norm": True},
        ]

        strides = [1, 2]

        feature_h = FLAGS.image_height
        feature_w = FLAGS.image_width

        count_ = 0
        min_size = min(FLAGS.image_height, FLAGS.image_width)
        while min_size > 1:
            min_size = (min_size + 1) // 2
            count_ += 1
        assert (FLAGS.cnn_count <= count_, "FLAGS.cnn_count should be <= {}!".format(count_))

        # CNN part
        with tf.variable_scope('cnn'):
            x = self.inputs
            # create n = len(cnn_params) convolunion layers
            for i in range(len(cnn_params)):
                with tf.variable_scope('unit-%d' % (i + 1)):

                    if i < len(cnn_params) - 1:
                        out = cnn_params[i + 1]["filters"]
                    else:
                        out = FLAGS.out_channels

                    x = self._conv2d(x, 'cnn-%d' % (i + 1), cnn_params[i]["k_size"], cnn_params[i]["filters"], out, strides[0])
                    x = self._leaky_relu(x, FLAGS.leakiness)
                    if cnn_params[i]["use_pool"]:
                        # convert features maps for correct sequences for RNN based cells
                        # use only in final layers
                        if i == 3:
                            x = tf.layers.max_pooling2d(x, [3, 2], [3, 2],
                                                    padding='valid',
                                                    name='pool-%d' % (i + 1))
                        elif i == 5:
                            x = tf.layers.max_pooling2d(x, [3, 2], [3, 2],
                                                        padding='valid',
                                                        name='poolfinal')
                        else:
                            x = self._max_pool(x, 2, strides[1])
                    _, feature_h, feature_w, _ = x.get_shape().as_list()
            # just to see convolution output dimensions
            print('\nfeature_h: {}, feature_w: {}'.format(feature_h, feature_w))

        # RNN (GRU) part - better results using GRU cells
        with tf.variable_scope('lstm'):
            x = tf.transpose(x, [0, 2, 1, 3])  # [batch_size, feature_w, feature_h, FLAGS.out_channels]
            # treat `feature_w` as max_timestep in GRU.
            x = tf.reshape(x, [FLAGS.batch_size, feature_w, feature_h * FLAGS.out_channels])
            print('GRU input shape: {}'.format(x.get_shape().as_list()))
            self.seq_len = tf.fill([x.get_shape().as_list()[0]], feature_w)

            cell = tf.nn.rnn_cell.GRUCell(FLAGS.num_hidden)
            if self.mode == 'train':
                cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=FLAGS.output_keep_prob)

            cell1 = tf.nn.rnn_cell.GRUCell(FLAGS.num_hidden)
            if self.mode == 'train':
                cell1 = tf.nn.rnn_cell.DropoutWrapper(cell=cell1, output_keep_prob=FLAGS.output_keep_prob)

            # Stacking rnn cells
            stack = tf.nn.rnn_cell.MultiRNNCell([cell, cell1], state_is_tuple=True)
            initial_state = stack.zero_state(FLAGS.batch_size, dtype=tf.float32)

            # The second output is the last state and we will not use that
            outputs, _ = tf.nn.dynamic_rnn(
                cell=stack,
                inputs=x,
                sequence_length=self.seq_len,
                initial_state=initial_state,
                dtype=tf.float32,
                time_major=False
            )  # [batch_size, max_stepsize, FLAGS.num_hidden]

            # Reshaping to apply the same weights over the timesteps
            outputs = tf.reshape(outputs, [-1, FLAGS.num_hidden])  # [batch_size * max_stepsize, FLAGS.num_hidden]

            W = tf.get_variable(name='W_out',
                                shape=[FLAGS.num_hidden, num_classes],
                                dtype=tf.float32,
                                initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer
            b = tf.get_variable(name='b_out',
                                shape=[num_classes],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            self.logits = tf.matmul(outputs, W) + b
            # Reshaping back to the original shape
            shape = tf.shape(x)
            self.logits = tf.reshape(self.logits, [shape[0], -1, num_classes])
            # print(self.logits)
            # Time major
            self.logits = tf.transpose(self.logits, (1, 0, 2))

    def _build_train_op(self):
        self.global_step = tf.train.get_or_create_global_step()

        self.loss = tf.nn.ctc_loss(labels=self.labels,
                                   inputs=self.logits,
                                   preprocess_collapse_repeated=False,
                                   ignore_longer_outputs_than_inputs=True,
                                   ctc_merge_repeated=False,
                                   sequence_length=self.seq_len)
        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar('cost', self.cost)

        self.lrn_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                   self.global_step,
                                                   FLAGS.decay_steps,
                                                   FLAGS.decay_rate,
                                                   staircase=True)
        tf.summary.scalar('learning_rate', self.lrn_rate)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lrn_rate,
                                                    beta1=FLAGS.beta1,
                                                    beta2=FLAGS.beta2).minimize(self.loss,
                                                                                global_step=self.global_step)
        train_ops = [self.optimizer] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)
        # this decoder is better than beam_search_decoder
        # decode sequences data from GRU cells
        self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.logits, self.seq_len, merge_repeated=False)
        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)

    def _conv2d(self, x, name, filter_size, in_channels, out_channels, strides):
        """
        basic function for creating convolution operation
        :param x: input tensor
        :param name: name of input tensor
        :param filter_size: filter matrix size
        :param in_channels: input channels (output from previous layer or image channel)
        :param out_channels: output channels
        :param strides: step for filter matrix
        :return: convolved data
        """
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='W',
                                     shape=[filter_size, filter_size, in_channels, out_channels],
                                     dtype=tf.float32,
                                     initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer

            b = tf.get_variable(name='b',
                                shape=[out_channels],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            con2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding='SAME')

        return tf.nn.bias_add(con2d_op, b)

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _max_pool(self, x, ksize, strides):
        """
        Pooling operation for convolution layer
        :param x:
        :param ksize:
        :param strides:
        :return: pooled tensor
        """
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, strides, strides, 1],
                              padding='SAME',
                              name='max_pool')
