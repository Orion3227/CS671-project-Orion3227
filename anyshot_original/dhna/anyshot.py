#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pdb
import data_handler
# import autoencoder
from scipy.io import loadmat
import scipy.io as sio
import argparse
import random
import datetime
import os
import sys
import math
fudge = 1e-30

now = datetime.datetime.now()
dist = tf.contrib.distributions
parser = argparse.ArgumentParser()
pi = math.pi

parser.add_argument('--batch_size', '-bs', type=int, default = 128)
parser.add_argument('--test_batch_size', '-tbs', type=int, default = 20)
parser.add_argument('--n_classes', '-ns', type=int, default = 50)
parser.add_argument('--a_dim', '-ad', type=int, default = 85)
parser.add_argument('--z_dim', '-zd', type=int, default = 85)
parser.add_argument('--p_dim', '-pd', type=int, default = 2048)
parser.add_argument('--s_dim', '-sd', type=int, default = 85)
parser.add_argument('--h1_dim', '-h1d', type=int, default = 85)
parser.add_argument('--h2_dim', '-h2d', type=int, default = 85)

parser.add_argument('--v_var_rate', '-vvarrate', type=float, default = 1.0)
parser.add_argument('--c_var_rate', '-cvarrate', type=float, default = 1.0)

parser.add_argument('--w_var_rate', '-wvarrate', type=float, default = 0.01)

parser.add_argument('--lr', '-lr', type=float, default = 0.0001)
parser.add_argument('--epochs', '-epochs', type=int, default = 5)
parser.add_argument('--topk', '-topk', type=int, default = 5)

parser.add_argument('--gpu_mem_usage', '-gmu', type=float, default = 0.2)

parser.add_argument('--exp', '-exp', type=data_handler.str2bool, default = True)

parser.add_argument('--shuffle', '-shuf', type=data_handler.str2bool, default = True)
parser.add_argument('--split', '-sp', type=data_handler.str2bool, default = True) # True means PS
parser.add_argument('--spath', '-spa', type=str, default = '/st1/dhna/zeroshot/result/') # True means PS

FLAGS = parser.parse_args()

# NOTE Result destination folder
if FLAGS.exp == True:
    FLAGS.spath = FLAGS.spath.replace(FLAGS.spath.split('/')[-2], 'exp_'+FLAGS.spath.split('/')[-2])
    print ("[INFO] This is experimental version, result will be sotred in ", FLAGS.spath)

# NOTE Refresh directory
dirs = [name for name in os.listdir(FLAGS.spath)]
num_dir = len([i for i in dirs if i.count('LOG') == 1])
FLAGS.spath += ( "LOG_" +str(num_dir+1) + "_" + now.strftime('%Y-%m-%d') + "_" + str(FLAGS.lr))
os.mkdir(FLAGS.spath, 0777)

# NOTE TF Setting
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_mem_usage)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

x = tf.placeholder(tf.float32, shape=[None, FLAGS.p_dim])
y_ = tf.placeholder(tf.int32, shape=[None]) # 10 classes output
n_ = tf.placeholder(tf.int32, shape=[FLAGS.n_classes])
n = np.zeros([1, FLAGS.n_classes])

embed = tf.placeholder(tf.float32, shape=[FLAGS.n_classes, FLAGS.s_dim]) # 10 classes output
embed_test = tf.placeholder(tf.float32, shape=[FLAGS.n_classes, FLAGS.s_dim]) # 10 classes output
mask = tf.placeholder(tf.float32, shape = [None, FLAGS.n_classes])
mask_c = tf.placeholder(tf.float32, shape = [None, FLAGS.n_classes])
mask_c_inv = tf.placeholder(tf.float32, shape = [None, FLAGS.n_classes])

# NOTE Data Preprocessing
data_whole, file_whole = data_handler.get_data_from_weights('./xlsa17/data/AWA1/res101.mat')
#data_whole, label_whole = data_handler.vgg_get_data_from_weights('./vgg_19_awa.mat')
loaded_data = loadmat('./xlsa17/data/AWA1/att_splits.mat')
embed_ = np.transpose(loaded_data['att'].astype(float)) # check size and transpose to 50 * 85
#embed_ = loadmat('../data/binary_attr.mat')['binary_attr'].astype(float)

# NOTE Attribute Space Standardization
mean = np.mean(embed_, axis=1)
std = np.std(embed_, axis=1)
embed_ = (embed_ - np.expand_dims(mean, -1))/np.expand_dims(std, -1)

list_name = []
for k in range(len(loaded_data['allclasses_names'])):
    list_name.append(loaded_data['allclasses_names'][k][0][0].split('.')[-1])

train_list = loaded_data['trainval_loc']
val_list = loaded_data['test_seen_loc']
test_list = loaded_data['test_unseen_loc']

# NOTE Select dataset type
if FLAGS.split == True:
    print ("[INFO] ps split")
    data_train, label_train = data_handler.data_preprocessing(data_whole, train_list, file_whole, list_name)
    data_val, label_val = data_handler.data_preprocessing(data_whole, val_list, file_whole, list_name)
    data_test, label_test = data_handler.data_preprocessing(data_whole, test_list, file_whole,list_name)
else:
    print ("[INFO] ss split")
    list_whole = np.concatenate((np.concatenate((train_list, val_list), axis=0),test_list), axis=0)
    ss_awa_zsl = [5, 13, 14, 17, 23, 24, 33, 38, 41, 47] #SS Split ZSL Classes for AwA dataset
    #data_train, label_train, data_test, label_test = data_handler.ss_data_preprocessing(data_whole, list_whole.tolist(), file_whole, list_name, ss_awa_zsl)
    data_train, label_train, data_test, label_test = data_handler.vgg_ss_data_preprocessing(data_whole, label_whole, ss_awa_zsl)

# NOTE Shuffle
if FLAGS.shuffle == True:
    print ("[INFO] Suffling Data")
    comb_train = list(zip(data_train, label_train))
    comb_test = list(zip(data_test, label_test))
    comb_val = list(zip(data_val, label_val))

    random.shuffle(comb_train)
    random.shuffle(comb_test)
    random.shuffle(comb_val)

    data_train[:], label_train[:] = zip(*comb_train)
    data_test[:], label_test[:] = zip(*comb_test)
    data_val[:], label_val[:] = zip(*comb_val)

n_data = len(label_train)
sup_list = data_handler.get_zsl_list(FLAGS, label_train)
zsl_list = data_handler.get_zsl_list(FLAGS, label_test)

# NOTE Handle Z
z_s = build_z(x, FLAGS) # out: batch, z_dim

# NOTE Handle V
v_mean_temp = [tf.nn.relu(z.mean()) for z in z_s] # out: batch, z_dim
v_var_temp = [tf.nn.relu(z.var()) for z in z_s] # out: batch, z_dim
v_s_temp = [tf.nn.relu(z.sample()) for z in z_s] # out: batch, z_dim

# NOTE Update V
v_mean= tf.zeros([FLAGS.batch_size, FLAGS.z_dim])
v_var = tf.zeros([FLAGS.batch_size, FLAGS.z_dim])

v_mean = v_mean + tf.nn.embedding_lookup(v_mean_temp, y_) #  check dim
v_var = v_var + tf.nn.embedding_lookup(v_var_temp, y_)

# NOTE Handle C
c_loc = tf.divide(v_mean, n_) # check if class is empty, then it is filled with zeros
c_scale = tf.divide(v_var, n_) # check if class is empty, then it is filled with zeros

c = dist.Normal(loc=c_loc, scale=FLAGS.c_var_rate*tf.nn.softplus(c_scale)) # N param. needs to be added
prob_c = c.prob(x) # check

# NOTE Handle Loss
log_softmax = tf.reduce_sum(tf.multiply(prob_c, mask_c), -1) - tf.log(tf.reduce_sum(tf.multiply(tf.exp(prob_c), mask), -1) + fudge)
kl_c = tf.reduce_sum(dist.kl_divergence(c, dist.Normal(loc=tf.zeros([dimension]), scale=tf.ones([dimension]))))
kl_z = tf.reduce_sum(dist.kl_divergence(z_s, dist.Normal(loc=tf.zeros(                                                                                                            [dimension]), scale=tf.ones([dimension])))) #Check dimension

loss = tf.reduce_mean(tf.reduce_sum((log_softmax - kl_c), -1) - kl_z) # not sure

# TEST Phase
distance_embed = tf.argmin(tf.sqrt(tf.reduce_mean(tf.square(tf.reshape(prob_v.mean(), [FLAGS.batch_size, 1, FLAGS.z_dim]) - tf.reshape(embed, [1, FLAGS.n_classes, FLAGS.z_dim])), -1)), axis=-1)
pred_embed_accu = tf.reduce_mean(tf.cast(tf.equal(tf.to_int32(distance_embed), y_), 'float'))

pred_idx_gzsl = tf.argmin(distance_test, axis = -1)
pred_idx_zsl = tf.argmin(tf.multiply(distance_test, mask), axis = -1)
pred_gzsl = tf.reduce_mean(tf.cast(tf.equal(tf.to_int32(pred_idx_gzsl), y_), 'float'))
pred_zsl = tf.reduce_mean(tf.cast(tf.equal(tf.to_int32(pred_idx_zsl), y_), 'float'))

# NOTE tensorboard
train_writer = tf.summary.FileWriter(FLAGS.spath + "/train_log")
test_writer = tf.summary.FileWriter(FLAGS.spath + "/test_log")

#Visualize Phase
w_samp_mu = tf.matmul(x, w)

v_samp_vis = g_c
v_samp_mu = g_all
v_proto = g_all

# optim
global_step = tf.Variable(0, trainable=False)
train_step = tf.train.AdamOptimizer(FLAGS.lr).minimize(-loss, global_step = global_step) #need to maximize, so make negative
train_iter = int(len(label_train)/FLAGS.batch_size)
val_iter = int(len(label_val)/FLAGS.batch_size)
test_iter = int(len(label_test)/FLAGS.batch_size)

print ("[TRAIN] total iterations ", train_iter * FLAGS.epochs)
batch_size = FLAGS.batch_size
test_batch_size = FLAGS.batch_size


sess.run(tf.global_variables_initializer())
data_samp = data_val + data_test
label_samp = label_val + label_test

data_proto = data_train + data_val + data_test
label_proto = label_train + label_val + label_test
proto = data_handler.proto_generator(data_proto, label_proto, FLAGS.n_classes, 2) # last param means number of instances to use for making prototype
proto_global = proto
for epoch in range(FLAGS.epochs):
    for idx_batch in range(train_iter):
        batch_x = data_train[idx_batch*batch_size:idx_batch*batch_size + batch_size]
        batch_y_temp = label_train[idx_batch*batch_size:idx_batch*batch_size + batch_size]
        batch_y = np.zeros([FLAGS.batch_size, FLAGS.n_classes])
        batch_y[np.arange(FLAGS.batch_size), batch_y_temp] = 1 # One-hot Vector

        mask_ = np.ones([FLAGS.batch_size, FLAGS.n_classes])
        zsl_list_temp = np.insert(np.asarray([zsl_list,]*FLAGS.batch_size), [0], np.split(np.asarray(batch_y_temp), FLAGS.batch_size), axis=1)

        mask_[:,zsl_list]=0

        mask_class, mask_class_inv = data_handler.mask_maker(FLAGS,batch_y_temp)

        _, gstep, loss_, likelihood_ = sess.run([train_step, global_step, loss, likelihood], feed_dict = {x: batch_x, y:batch_y, y_:batch_y_temp, test_y_:np.arange(FLAGS.n_classes), embed:embed_, mask:mask_, mask_c:mask_class, mask_c_inv: mask_class_inv})

        if idx_batch%1000 == 0:
            print ("[Train] Iteration ", epoch*train_iter + idx_batch, "/", train_iter * FLAGS.epochs, "   *loss ", -loss_, "   *likeli ", -likelihood_)


    # NOTE Val Session
    val_corr = 0
    for idx_val_batch in range(val_iter):
        batch_val_x = data_val[idx_val_batch*batch_size:idx_val_batch*batch_size + batch_size]
        batch_val_y = label_val[idx_val_batch*batch_size:idx_val_batch*batch_size + batch_size]

        mask_class, mask_class_inv = data_handler.mask_maker(FLAGS,batch_val_y)

        pred_val_ = sess.run([pred_gzsl], feed_dict = {x: batch_val_x, y_: batch_val_y, test_y_:np.arange(FLAGS.n_classes), embed:embed_, mask_c:mask_class, mask_c_inv: mask_class_inv})

        val_corr = val_corr + np.sum(pred_val_)

    print ("[VAL] RESULT FOR BATCH ", epoch, "    [VAL] corr %.4f"%(val_corr/int(val_iter)))

    # NOTE Test Session
    zsl_corr = gzsl_corr = 0
    for idx_test_batch in range(test_iter):
        batch_test_x = data_test[idx_test_batch*test_batch_size:idx_test_batch*test_batch_size + test_batch_size]
        batch_test_y = label_test[idx_test_batch*test_batch_size:idx_test_batch*test_batch_size + test_batch_size]

        mask_ = np.ones([FLAGS.batch_size, FLAGS.n_classes], dtype=np.float32)*999
        mask_[:,zsl_list] = 1.

        mask_class, mask_class_inv = data_handler.mask_maker(FLAGS,batch_test_y)

        gstep, pred_zsl_, pred_gzsl_ = sess.run([global_step, pred_zsl, pred_gzsl,], feed_dict = {x: batch_test_x, y_: batch_test_y, test_y_:np.arange(FLAGS.n_classes), embed:embed_, mask:mask_, mask_c:mask_class, mask_c_inv: mask_class_inv})

        zsl_corr = zsl_corr + np.sum(pred_zsl_)
        gzsl_corr = gzsl_corr + np.sum(pred_gzsl_)

    print ("[TEST] RESULT FOR BATCH ", epoch, "    [ZSL] pred %.4f,    [GZSL] pred %.4f"%(zsl_corr/int(test_iter), gzsl_corr/int(test_iter)))
    print ()
    #Save for Visualization
    if epoch % 5 == 0:
        batch_samp_train_x, batch_samp_train_y = data_handler.vis_sampler(data_train, label_train, FLAGS.n_classes, 10)
        samp_train_iter = int(len(batch_samp_train_y)/FLAGS.batch_size)

        train_loss = []
        train_stddev = []
        samp_ground_truth = []
        for idx_test_batch in range(samp_train_iter):
            batch_test_x = batch_samp_train_x[idx_test_batch*test_batch_size:idx_test_batch*test_batch_size + test_batch_size]
            batch_test_y = batch_samp_train_y[idx_test_batch*test_batch_size:idx_test_batch*test_batch_size + test_batch_size]

            mask_ = np.ones([FLAGS.batch_size, FLAGS.n_classes], dtype=np.float32)*999
            mask_[:,zsl_list] = 1.

            mask_class, mask_class_inv = data_handler.mask_maker(FLAGS,batch_test_y)

            v_stddev_, loss_ = sess.run([v_var, likelihood], feed_dict = {x: batch_test_x, y_: batch_test_y, embed:embed_, mask:mask_, mask_c:mask_class, mask_c_inv: mask_class_inv})
            train_loss.append(loss_)
            train_stddev.append(v_stddev_)
            samp_ground_truth.append(batch_test_y)


        batch_samp_x, batch_samp_y = data_handler.vis_sampler(data_samp, label_samp, FLAGS.n_classes, 10)
        samp_iter = int(len(batch_samp_y)/FLAGS.batch_size)

        distance_embed__ = []
        prediction_zsl = []
        prediction_gzsl = []
        ground_truth = []
        ground_truth_samp = []
        w_samp__ = []
        w_samp_mu__ = []
        v_samp__ = []
        v_samp_mu__ = []
        v_samp_stddev_ = []
        embed_accu_ = 0

        for idx_test_batch in range(samp_iter):
            batch_test_x = batch_samp_x[idx_test_batch*test_batch_size:idx_test_batch*test_batch_size + test_batch_size]
            batch_test_y = batch_samp_y[idx_test_batch*test_batch_size:idx_test_batch*test_batch_size + test_batch_size]

            mask_ = np.ones([FLAGS.batch_size, FLAGS.n_classes], dtype=np.float32)*999
            mask_[:,zsl_list] = 1.

            mask_class, mask_class_inv = data_handler.mask_maker(FLAGS,batch_test_y)

            v_stddev_, w_samp_mu_, embed_corr, distance_embed_, v_samp_mu_, pred_idx_zsl_, pred_idx_gzsl_ = sess.run([v_var, w_samp_mu, pred_embed_accu, distance_embed, v_samp_mu, pred_idx_zsl, pred_idx_gzsl], feed_dict = {x: batch_test_x, y_: batch_test_y, embed:embed_, mask:mask_, mask_c:mask_class, mask_c_inv: mask_class_inv})
            prediction_zsl.append(np.squeeze(pred_idx_zsl_))
            prediction_gzsl.append(np.squeeze(pred_idx_gzsl_))
            ground_truth.append(np.squeeze(batch_test_y))
            v_samp_mu__.append(np.squeeze(v_samp_mu_))
            w_samp_mu__.append(np.squeeze(w_samp_mu_))
            v_samp_stddev_.append(np.squeeze(v_stddev_))
            distance_embed__.append(np.squeeze(distance_embed_))

            for k in range(3):
                v_samp_vis_= sess.run([v_samp_vis], feed_dict = {x: batch_test_x, y_: batch_test_y, embed:embed_, mask:mask_, mask_c:mask_class, mask_c_inv: mask_class_inv})
                v_samp__.append(np.squeeze(v_samp_vis_))
                ground_truth_samp.append(np.squeeze(batch_test_y))
            embed_accu_ = embed_accu_ + embed_corr

        print ("Embedding Accuracy ", float(embed_accu_)/samp_iter)

        sio.savemat(FLAGS.spath + "/visual_log_epoch_%d.mat"%(epoch), {'samp_ground_truth':samp_ground_truth, 'train_stddev': train_stddev, 'train_loss': train_loss, 'dist_embed': distance_embed__, 'w_samp_vis':w_samp__, 'w_samp_mu': w_samp_mu__,'v_samp_vis':v_samp__, 'v_samp_mu':v_samp_mu__, 'v_samp_stddev': v_samp_stddev_, 'samp_label':batch_samp_y, 'pred_zsl': prediction_zsl, 'pred_gzsl': prediction_gzsl, 'gt': ground_truth, 'gt_samp': ground_truth_samp})
