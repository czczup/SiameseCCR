import tensorflow as tf


class Siamese(object):
    def __init__(self):
        with tf.name_scope("input"):
            self.left = tf.placeholder(tf.float32, [None, 44, 44, 1], name='left')
            self.right = tf.placeholder(tf.float32, [None, 44, 44, 1], name='right')
            label = tf.placeholder(tf.int32, [None, 1], name='label')  # 1 if same, 0 if different
            self.label = tf.to_float(label)
        self.training = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        with tf.variable_scope("Siamese") as scope:
            self.left_output = self.model(self.left)
            scope.reuse_variables()
            self.right_output = self.model(self.right)
        self.prediction, self.loss, self.test_param = self.contrastive_loss(self.left_output, self.right_output, self.label)
        tf.summary.scalar('loss', self.loss)
        self.batch_size = 512
        self.learning_rate = tf.train.exponential_decay(1e-4, self.global_step, decay_steps=6E5//self.batch_size,
                                                   decay_rate=0.98, staircase=True)
        tf.summary.scalar('learning_rate', self.learning_rate)
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.less(self.prediction, 0.5), tf.less(self.label, 0.5))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def conv2d(self, x, output_filters, kernel, strides=1, padding="SAME"):
        conv = tf.contrib.layers.conv2d(x, output_filters, [kernel, kernel], activation_fn=tf.nn.relu, padding=padding,
                                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        stride=strides)
        return conv

    def residual(self, x, num_filters, kernel, strides, training=False, with_shortcut=False):
        with tf.name_scope("residual"):
            conv1 = self.conv2d(x, num_filters[1], kernel=1, strides=strides)
            bn1 = tf.layers.batch_normalization(conv1, training=self.training)
            relu1 = tf.nn.relu(bn1)
            conv2 = self.conv2d(relu1, num_filters[2], kernel=3)
            bn2 = tf.layers.batch_normalization(conv2, training=self.training)
            relu2 = tf.nn.relu(bn2)
            conv3 = self.conv2d(relu2, num_filters[3], kernel=1)
            bn3 = tf.layers.batch_normalization(conv3, training=self.training)
            if with_shortcut:
                shortcut = self.conv2d(x, num_filters[3], kernel=1, strides=strides)
                bn_shortcut = tf.layers.batch_normalization(shortcut, training=self.training)
                residual = tf.nn.relu(bn_shortcut + bn3)
            else:
                residual = tf.nn.relu(x + bn3)
            return residual

    def model(self, x, reuse=False):
        # channel = int(input("please input channel number:"))
        channel = 64
        with tf.variable_scope("conv1") as scope:
            conv1 = self.conv2d(x, channel, 7, 1)
            bn = tf.layers.batch_normalization(conv1, training=self.training)
            relu = tf.nn.relu(bn)
            pool = tf.nn.max_pool(relu, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME")
        with tf.variable_scope("block1") as scope:
            res1 = self.residual(pool, [channel, channel//2, channel//2, channel*2], 3, 1, with_shortcut=True)
            res2 = self.residual(res1, [channel*2, channel//2, channel//2, channel*2], 3, 1)
            print(res2)
        with tf.variable_scope("block2") as scope:
            res3 = self.residual(res2, [channel*2, channel, channel, channel*4], 3, 2, with_shortcut=True)
            res4 = self.residual(res3, [channel*4, channel, channel, channel*4], 3, 1)
            print(res4)
        with tf.variable_scope("block3") as scope:
            res5 = self.residual(res4, [channel*4, channel*2, channel*2, channel*8], 3, 2, with_shortcut=True)
            res6 = self.residual(res5, [channel*8, channel*2, channel*2, channel*8], 3, 1)
            print(res6)
        with tf.variable_scope("block4") as scope:
            res7 = self.residual(res6, [channel*8, channel*4, channel*4, channel*16], 3, 2, with_shortcut=True)
            res8 = self.residual(res7, [channel*16, channel*4, channel*4, channel*16], 3, 1)
            print(res8)
            pool = tf.nn.avg_pool(res8, [1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')
            flatten = tf.layers.flatten(pool)  # 2*2*1024=4096
            print(flatten)
        with tf.variable_scope("fc1") as scope:
            hidden_Weights1 = tf.Variable(tf.truncated_normal([channel*16, 256], stddev=0.1))  # 45-7040 40-5632
            hidden_biases1 = tf.Variable(tf.constant(0.1, shape=[256]))
            net = tf.add(tf.matmul(flatten, hidden_Weights1), hidden_biases1)
        return net

    def contrastive_loss(self, model1, model2, y):
        with tf.name_scope("output"):
            output_difference = tf.abs(model1 - model2)
            W = tf.Variable(tf.random_normal([256, 1], stddev=0.1), name='W')
            b = tf.Variable(tf.zeros([1, 1]) + 0.1, name='b')
            wx_plus_b = tf.matmul(output_difference, W) + b
            y_ = tf.nn.sigmoid(wx_plus_b, name='distance')
            # y_ = wx_plus_b
        # Calculate mean loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=wx_plus_b, labels=y)
            loss = tf.reduce_mean(losses)
        return y_, loss, [W, b]


if __name__ == '__main__':
    model = Siamese()
    # var_list = tf.global_variables()
    # for var in var_list:
    #     if "batch" not in var.name:
    #         print(var)

