"""
CUDA_VISIBLE_DEVICES=0 python -i 101_Training.py \
--lable_processed True \

"""
import tensorflow as tf
import os, random
import data_loader
import numpy as np
import pdb



class CNN_model():
	def __init__(self, sess, config, name):
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
		self.conv = self.input_layer

		for n_block in range(self.num_block):
			# Convolutional Layer
			for n_layer in range(3):
				self.conv = tf.layers.conv2d(inputs=self.conv, filters=self.filt[n_block],
					kernel_size=[self.window, self.window], padding="same", activation=tf.nn.relu)

			# Pooling Layer
			self.conv = tf.layers.max_pooling2d(inputs=self.conv, pool_size=[2, 2], strides=2)

		# Dense Layer
		self.pool_flat = tf.reshape(self.conv, [-1, int(self.height * self.width / (2**self.num_block)**2) * self.filt[self.num_block-1]])
		self.dense = tf.layers.dense(inputs=self.pool_flat, units=512, activation=tf.nn.relu)
		#dropout = tf.layers.dropout(inputs=dense, rate=0.4)

		# Logits Layer
		self.logits = tf.layers.dense(inputs=self.dense, units=self.n_classes)
		#pdb.set_trace()

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

