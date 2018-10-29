import tensorflow as tf
import os, random
import data_loader
import numpy as np
import pdb
import tensorflow.contrib.layers as layers


class ResNet:
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

		self.X = tf.placeholder(tf.float32, [None, self.width, self.height, self.channels])
		self.Y = tf.placeholder(tf.float32, [None, self.n_classes])
		Reshaped = tf.reshape(self.X, [-1, self.width, self.height, self.channels])

		conv = layers.conv2d(inputs=Reshaped, num_outputs=16, kernel_size=3, stride=1, activation_fn=self.selu)
		Residual_flow = self.resnet_block(conv, 32, stack=False)
		Residual_flow = self.resnet_block(Residual_flow, 64)
		Residual_flow = self.resnet_block(Residual_flow, 64)

		Avg_Pool = layers.avg_pool2d(inputs=Residual_flow, kernel_size=2, stride=2)
		Flatten = layers.flatten(Avg_Pool)
		self.logits = layers.fully_connected(inputs=Flatten, num_outputs=self.n_classes, activation_fn=None)

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

	def selu(self, x):
		alpha = 1.6732632423543772848170429916717
		scale = 1.0507009873554804934193349852946
		return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)
		
	def resnet_block(self, input_data, layers_num, stack=True):	
		if(stack==True): stride = 1		
		else: stride = 1
		Conv1 = layers.conv2d(inputs=input_data, num_outputs=layers_num, kernel_size=3, stride=1, activation_fn=self.selu)
		Conv2 = layers.conv2d(inputs=Conv1, num_outputs=layers_num, kernel_size=3, stride=stride, activation_fn=None)

		if(stack==True): 
			input_data = layers.conv2d(inputs=input_data, num_outputs=layers_num, kernel_size=1, stride=stride, activation_fn=None)
		if int(input_data.shape[3]) != int(Conv2.shape[3]):
			input_data = tf.concat((input_data, input_data), axis = 3)
		return self.selu(input_data + Conv2)

