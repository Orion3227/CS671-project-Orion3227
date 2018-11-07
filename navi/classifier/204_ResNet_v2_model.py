import tensorflow as tf
import os, random
import data_loader
import numpy as np
import pdb
module = __import__('012_ResNet_ops')


class ResNet_v2:
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
		self.n = 5
		self.reuse = False

		self.X = tf.placeholder(tf.float32, [None, self.width, self.height, self.channels])
		self.Y = tf.placeholder(tf.float32, [None, self.n_classes])
		
		input_tensor_batch = self.X
		layers = []
		with tf.variable_scope('conv0', reuse=self.reuse):
			conv0 = module.conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)
			module.activation_summary(conv0)
			layers.append(conv0)

		for i in range(self.n):
			with tf.variable_scope('conv1_%d' %i, reuse=self.reuse):
				if i == 0:
					conv1 = module.residual_block(layers[-1], 16, first_block=True)
				else:
					conv1 = module.residual_block(layers[-1], 16)
				module.activation_summary(conv1)
				layers.append(conv1)

		for i in range(self.n):
			with tf.variable_scope('conv2_%d' %i, reuse=self.reuse):
				conv2 = module.residual_block(layers[-1], 32)
				module.activation_summary(conv2)
				layers.append(conv2)

		for i in range(self.n):
			with tf.variable_scope('conv3_%d' %i, reuse=self.reuse):
				conv3 = module.residual_block(layers[-1], 64)
				layers.append(conv3)
			# pdb.set_trace()
			assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

		with tf.variable_scope('fc', reuse=self.reuse):
			in_channel = layers[-1].get_shape().as_list()[-1]
			bn_layer = module.batch_normalization_layer(layers[-1], in_channel)
			relu_layer = tf.nn.relu(bn_layer)
			global_pool = tf.reduce_mean(relu_layer, [1, 2])

			assert global_pool.get_shape().as_list()[-1:] == [64]
			output = module.output_layer(global_pool, 10)
			layers.append(output)

		self.logits = layers[-1]
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



