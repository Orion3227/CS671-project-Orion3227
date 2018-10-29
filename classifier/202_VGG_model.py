"""
CUDA_VISIBLE_DEVICES=0 python -i 101_Training.py \
--lable_processed True \

"""
import tensorflow as tf
import os, random
import data_loader
import numpy as np
import pdb
L = __import__('302_VGG_module')

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
	def __init__(self, sess, config, name, vgg19_npy_path=None):
		self.sess = sess
		self.name = name
		self._build_net(config)
		self.training = tf.placeholder(tf.bool)
		

	def _build_net(self, config):

		self.window = 3
		self.height = config.im_size
		self.width = config.im_size
		self.channels = 3
		self.n_classes = config.n_classes
		self.num_block = int(np.log(config.im_size/7)/np.log(2))
		self.filt = [32, 64, 128, 256, 256]
		self.im_size = [self.height, self.width, self.channels]

		self.X = tf.placeholder(tf.float32, [None, self.height, self.width, self.channels])
		self.Y = tf.placeholder(tf.float32, [None, self.n_classes])

		# -------------------- Model -------------------- #
		self.input_layer = tf.reshape(self.X, [-1, self.height, self.width, self.channels])
		# assuming 224x224x3 input_tensor

		# define image mean
		rgb_mean = np.array([116.779, 123.68, 103.939], dtype=np.float32)
		mu = tf.constant(rgb_mean, name="rgb_mean")
		keep_prob = 0.5

		# subtract image mean
		net = tf.subtract(self.input_layer*255, mu, name="input_mean_centered")

		# block 1 -- outputs 112x112x64
		net = L.conv(net, name="conv1_1", kh=3, kw=3, n_out=64)
		net = L.conv(net, name="conv1_2", kh=3, kw=3, n_out=64)
		net = L.pool(net, name="pool1", kh=2, kw=2, dw=2, dh=2)

		# block 2 -- outputs 56x56x128
		net = L.conv(net, name="conv2_1", kh=3, kw=3, n_out=128)
		net = L.conv(net, name="conv2_2", kh=3, kw=3, n_out=128)
		net = L.pool(net, name="pool2", kh=2, kw=2, dh=2, dw=2)

		# # block 3 -- outputs 28x28x256
		net = L.conv(net, name="conv3_1", kh=3, kw=3, n_out=256)
		net = L.conv(net, name="conv3_2", kh=3, kw=3, n_out=256)
		net = L.pool(net, name="pool3", kh=2, kw=2, dh=2, dw=2)

		# block 4 -- outputs 14x14x512
		net = L.conv(net, name="conv4_1", kh=3, kw=3, n_out=512)
		net = L.conv(net, name="conv4_2", kh=3, kw=3, n_out=512)
		net = L.conv(net, name="conv4_3", kh=3, kw=3, n_out=512)
		net = L.pool(net, name="pool4", kh=2, kw=2, dh=2, dw=2)

		# block 5 -- outputs 7x7x512
		net = L.conv(net, name="conv5_1", kh=3, kw=3, n_out=512)
		net = L.conv(net, name="conv5_2", kh=3, kw=3, n_out=512)
		net = L.conv(net, name="conv5_3", kh=3, kw=3, n_out=512)
		net = L.pool(net, name="pool5", kh=2, kw=2, dw=2, dh=2)

		# flatten
		flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
		net = tf.reshape(net, [-1, flattened_shape], name="flatten")

		# fully connected
		net = L.fully_connected(net, name="fc6", n_out=4096)
		net = tf.nn.dropout(net, keep_prob)
		net = L.fully_connected(net, name="fc7", n_out=4096)
		net = tf.nn.dropout(net, keep_prob)
		self.net = net/255
		self.logits = L.fully_connected(self.net, name="fc8", n_out=self.n_classes)

		# self.prob = tf.nn.softmax(self.fc8, name="prob")

		# -------------------- Objective -------------------- #

		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			logits=self.logits, labels=self.Y), name='Loss')
		total_var = tf.global_variables() 
		self.optimizer = tf.train.AdamOptimizer(0.001, epsilon=0.01).minimize(self.cost)
		self.is_correct = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))

		# accuracy = 1 - tf.reduce_mean(tf.abs(tf.round(tf.nn.sigmoid(self.logits)) - tf.round(Y)))
		# accuracy = 1 - tf.reduce_mean(tf.abs(tf.round(self.logits) - tf.round(Y)))
		self.accuracy = tf.reduce_mean(tf.cast(self.is_correct, tf.float32))

		self.writer = tf.summary.FileWriter("./board/sample", self.sess.graph)
		self.acc_hist = tf.summary.scalar("Training_accuracy", self.accuracy)
		self.merged = tf.summary.merge_all()

		init = tf.global_variables_initializer()
		self.sess.run(init)
		self.saver = tf.train.Saver(total_var)

		tf.train.start_queue_runners(sess=self.sess)

	def train(self, x_data, y_data, training=True):
		return self.sess.run([self.optimizer, self.cost, self.merged, self.accuracy], feed_dict={self.X: x_data, self.Y: y_data, self.training: training})

	def get_accuracy(self, x_test, y_test, training=False):
		return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})

	def predict(self, x_data, training=False):
		return self.sess.run(tf.argmax(self.logits, 1), feed_dict={self.X: x_data, self.training: training})


