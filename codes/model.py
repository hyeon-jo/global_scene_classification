import tensorflow as tf
import cv2

class cnn_model():
    def __init__(self, name):
        label_size = 3
        learning_rate = 0.1
        img_width = 256
        img_height = 256
        self.name = name

        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, img_width, img_height, 3])
            img_tensor = self.X
            self.Y = tf.placeholder(tf.float32, [None, 3])

            W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
            conv1 = tf.nn.conv2d(img_tensor, W1, strides = [1, 1, 1, 1], padding='SAME')
            relu1 = tf.nn.relu(conv1)
            pool1 = tf.nn.avg_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
            conv2 = tf.nn.conv2d(pool1, W2, strides=[1, 1, 1, 1], padding='SAME')
            relu2 = tf.nn.relu(conv2)
            pool2 = tf.nn.avg_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

            W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
            conv3 = tf.nn.conv2d(pool2, W3, strides=[1, 1, 1, 1], padding='SAME')
            relu3 = tf.nn.relu(conv3)
            pool3 = tf.nn.avg_pool(relu3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

            flatten = tf.reshape(pool3, [-1, 128 * 4 * 4])

            W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 512], initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([512]))
            relu4 = tf.nn.relu(tf.matmul(flatten, W4) + b4)

            W5 = tf.get_variable("W5", shape=[512, label_size], initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([label_size]))
            self.logits = tf.matmul(relu4, W5) + b5
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.Y))
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(self.cost)

    def train(self, x_data, y_data):
        return self.sess.run(self.logits, feed_dict={self.X: x_data, self.Y: y_data})

    def predict(self, x_data):
        return self.sess.run(self.logits, feed_dict={self.X: x_data})