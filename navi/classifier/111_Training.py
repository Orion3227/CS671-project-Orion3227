"""
CUDA_VISIBLE_DEVICES=0 python -i 111_Training.py \
--data_path=/home/siit/navi/data/input_data/cifar/ \
--meta_path=/home/siit/navi/data/meta_data/cifar/ \
--n_classes=10 --im_size=224 --batch_size=10 \
--label_processed True \

"""
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='/home/siit/navi/data/input_data/mnist_png/')
parser.add_argument('--meta_path', type=str, dest='meta_path', default='/home/siit/navi/data/meta_data/mnist_png/')
parser.add_argument('--model_path', type=str, dest='model_path', default='/shared/data/models/')
parser.add_argument('--epoch', type=int, dest='epoch', default=1000)

parser.add_argument('--n_classes', type=int, dest='n_classes', default=10)
parser.add_argument('--im_size', type=int, dest='im_size', default=28)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=100)

parser.add_argument('--label_processed', type=bool, dest='label_processed', default=True)
parser.add_argument('--save_freq', type=int, dest='save_freq', default=1000)
parser.add_argument('--print_freq', type=int, dest='print_freq', default=50)
parser.add_argument('--memory_usage', type=float, dest='memory_usage', default=0.96)

parser.add_argument('--mode', type=str, dest='mode', default='pretrained')
parser.add_argument('--load_checkpoint', type=bool, dest='load_checkpoint', default=False)
parser.add_argument('--checkpoint_path', type=str, dest='checkpoint_path', default='./checkpoints')
config, unparsed = parser.parse_known_args() 

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.memory_usage)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

import os, random
import data_loader
import numpy as np
import pdb


# -------------------- Model -------------------- #
depth = 3
window = 3
height = config.im_size
width = config.im_size
channels = 3
im_size = [height, width, channels]

# model = __import__('201_CNN_model').CNN_model(sess, config, 'CNN_model')
# model = __import__('202_VGG_model').Vgg19(sess, config, 'VGG_19')
# model = __import__('203_ResNet_model').ResNet(sess, config, 'ResNet')
model = __import__('204_ResNet_v2_model').ResNet_v2(sess, config, 'ResNet')

# -------------------- Data maniging -------------------- #

#label_file = np.load(os.path.join('/shared/data/celeb_cartoon/','attributes.npz'))
label_file = np.load(os.path.join(config.meta_path, 'path_label_dict.npy'))

list_files = [os.path.join(dp, f)
		for dp, dn, filenames in os.walk(config.meta_path) 
		for f in filenames if 'dict.npy' in f]
list_files.sort()

batch_size = config.batch_size


# -------------------- Training -------------------- #

# If you want to debug the model, write the following command on the console
# log_ = model.sess.run([model.logits], feed_dict={model.X: Xbatch, model.Y: Ybatch, model.training: True})

counter = 0
for epoch in range(config.epoch):
	for list_file in list_files:

		f = np.load(list_file)
		
		path_label_dict = f.item()
		input_file_list = list(path_label_dict.keys())
		# You should shuffle the list. 
		# The network will be stupid if you don't
		random.shuffle(input_file_list)
		all_data = [line for line in input_file_list]
		train_data = all_data[1000:]
		test_data = all_data[:1000]

		num_file = len(train_data)

		if num_file ==0:
			break

		# print('Number of input files: \t{}'.format(num_file))
		total_batch = int(num_file / batch_size)
		total_cost = 0
		final_acc = 0

		for i in range(total_batch):
			# Get the batch as [batch_size, 28,28] and [batch_size, n_classes] ndarray
			label_list = np.expand_dims(path_label_dict[train_data[i*batch_size]], axis = 0)
			for j in range(1, batch_size):
				label_list = np.concatenate((label_list, np.expand_dims(
					path_label_dict[train_data[i*batch_size + j]], 
					axis = 0)), axis = 0)
			Ybatch = np.reshape(label_list, [batch_size, config.n_classes])

			Xbatch = data_loader.queue_data_dict(
				train_data[i*batch_size:(i+1)*batch_size], im_size, config.label_processed)

			_, cost_val, acc, acc_ = model.sess.run(
				[model.optimizer, model.cost, model.merged, model.accuracy], 
				feed_dict={model.X: Xbatch, model.Y: Ybatch, model.training: True})

			total_cost += cost_val

			counter += 1

			if np.mod(counter, config.print_freq) == 0:
				print('Step:', '%05dk' % (counter),
					'\tAvg. cost =', '{:.5f}'.format(cost_val),
					'\tAcc: {:.5f}'.format(acc_))
				model.writer.add_summary(acc, counter)

			# Save the model
			if np.mod(counter, config.save_freq) == 0:
				if not os.path.exists(config.checkpoint_path):
					os.mkdir(config.checkpoint_path)
				model.saver.save(sess, os.path.join(config.checkpoint_path, 
					'vgg19_{0:03d}k'.format(int(counter/1000))))
				print('Model ')
	

# -------------------- Testing -------------------- #


Xbatch, Ybatch, _ = data_loader.queue_data(
	test_data, label_list, im_size)

accuracy_ = sess.run(accuracy, feed_dict = {X: Xbatch, Y: Ybatch})
print('Accuracy:', accuracy_)
