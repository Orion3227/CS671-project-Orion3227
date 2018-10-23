from __future__ import print_function
import tensorflow as tf
import numpy as np
import time, os, csv, random, pdb
from pylab import *

from utils.accumulator import Accumulator
from utils.train import *
from utils.mnist import mnist_subset
from utils.config import get_args, setup


args = get_args(sys.argv[1:])
savedir = setup(args)

# To protect Human Error!
if args.model == 'dpse':
    from model.vsepro.vsepro_nway_kl import combine
else:
    print ("select proper args.model!")
    raise NotImplementedError()

xtr, xtrte, xte, nxtr, nxtrte, nxte = mnist_subset([100]*5)

# set-embedding placeholder x 10
x_emb = [tf.placeholder(tf.float32, [None, 784], name='x_emb_%d'%k)
        for k in range(args.way)]

# query placeholder x 10
x_qry = [tf.placeholder(tf.float32, [None, 784], name='x_qry_%d'%k)
        for k in range(args.way)]

# query label placeholder x 10
y_qry = [tf.placeholder(tf.float32, [None, args.way], name='y_qry_%d'%k)
        for k in range(args.way)]

# Network to run
net = combine(args, x_emb, x_qry, y_qry, True)
tnet = combine(args, x_emb, x_qry, y_qry, False, reuse=True)

def data_queue(args, x, x_n, cidx, is_trte=False, xtrte=None):
    fd = {}
    for k in range(args.way):
        idx = cidx[k]

        #Shuffle the data of each task
        np.random.shuffle(x[idx])

        n = args.shot
        m = args.query

        #Support
        fd[x_emb[k]] = x[idx][:n]
        #Query
        fd[x_qry[k]] = x[idx][-m:]

        #One-hot encoding
        y = np.zeros(args.way)
        y[k] = 1
        y = np.tile(y, (m, 1)) #duplicate as much as query points
        fd[y_qry[k]] = y

    return fd

def train():
    import random

    loss = -net['elbo'] + net['proto_loss']*args.lamb

    global_step = tf.train.get_or_create_global_step()
    lr_step = args.n_trn_epsd/3
    lr = tf.train.piecewise_constant(tf.cast(global_step, tf.int32),
            [lr_step, lr_step*2], [1e-3, 1e-3*0.5, 1e-3*0.5*0.5])
    train_op = tf.train.AdamOptimizer(lr).minimize(loss,
            global_step=global_step)

    saver = tf.train.Saver(net['weights'])
    logfile = open(os.path.join(savedir, 'train.log'), 'w', 0)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # to_run
    train_logger = Accumulator('elbo', 'proto_loss')
    train_to_run = [train_op, net['elbo'], net['proto_loss']]
    for i in range(args.n_trn_epsd):
        #train feed_dict
        cidx = random.sample(xrange(len(nxtr)), args.way)
        fdtr = data_queue(args, xtr, nxtr, cidx)

        # train
        train_logger.clear()
        start = time.time()
        train_logger.accum(sess.run(train_to_run, feed_dict=fdtr))

        if i % 100 == 0:
            train_logger.print_(header='train', epoch=i+1,
                    time=time.time()-start, logfile=logfile)

            line = 'Epoch %d start, learning rate %f' % (i+1, sess.run(lr))
            print('\n' + line)
            logfile.write('\n' + line + '\n')

            accu = sess.run(net['acc'], feed_dict=fdtr)
            print ("test accu ", np.mean(accu))

    saver.save(sess, os.path.join(savedir, 'model'))

    logfile.close()
    saver.save(sess, os.path.join(savedir, 'model'))


def test():
    import random

    sess = tf.Session()
    saver = tf.train.Saver(tnet['weights'])
    saver.restore(sess, os.path.join(savedir, 'model'))

    test_logger = Accumulator('elbo', 'proto_loss', 'acc')
    test_to_run = [tnet['elbo'], tnet['proto_loss'], tnet['acc']]

    test_logger.clear()
    for i in range(args.n_tst_epsd):
        #train feed_dict
        cidx = random.sample(xrange(len(nxte)), args.way)
        fdte = data_queue(args, xte, nxte, cidx)

        # test
        test_logger.accum(sess.run(test_to_run, feed_dict=fdte))
        if (i+1) % 100 == 0:
            test_logger.print_(header='test', epoch=i+1)

if __name__=='__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        raise ValueError('Invalid mode %s' % args.mode)
