from tensorflow.examples.tutorials.mnist import input_data
from paths import MNIST_PATH
import numpy as np

K = 5

def mnist_input(batch_size):
    mnist = input_data.read_data_sets(MNIST_PATH, one_hot=True, validation_size=0)
    n_train_batches = mnist.train.num_examples/batch_size
    n_test_batches = mnist.test.num_examples/batch_size
    return mnist, n_train_batches, n_test_batches

def mnist_subset(nlist):
    mnist = input_data.read_data_sets(MNIST_PATH, one_hot=False, validation_size=0)
    x, y = mnist.train.images, mnist.train.labels

    #xtr = [x[y==k] for k in range(K)]
    xtr = [x[y==k][:nlist[k],:] for k in range(K)]
    xte_seen = [x[y==k][nlist[k]:2*nlist[k],:] for k in range(K)]
    xte_unseen = [x[y==k] for k in range(K,2*K)]
    #xte_unseen = [x[y==k][:nlist[k-K],:] for k in range(K,2*K)]
    return xtr, xte_seen, xte_unseen, nlist, nlist, nlist

def mnist_subset2(nlist):
    mnist = input_data.read_data_sets(MNIST_PATH, one_hot=True, validation_size=0)
    xtr, ytr = mnist.train.images, mnist.train.labels
    #xte, yte = mnist.test.images, mnist.test.labels

    ytr_ = np.argmax(ytr, 1)
    xtr_list = [xtr[ytr_==k][:nlist[k],:] for k in range(K)]
    ytr_list = [ytr[ytr_==k][:nlist[k],:K] for k in range(K)]

    #yte_ = np.argmax(yte, 1)
    #xte_list = [xte[yte_==k] for k in range(10)]
    #yte_list = [yte[yte_==k] for k in range(10)]
    xtrte_list = [xtr[ytr_==k][nlist[k]:nlist[k]*2,:] for k in range(K)]
    ytrte_list = [ytr[ytr_==k][nlist[k]:nlist[k]*2,:K] for k in range(K)]

    xte_list = [xtr[ytr_==k][:nlist[k-5],:] for k in range(5,10)]
    yte_list = [ytr[ytr_==k][:nlist[k-5],5:10] for k in range(5,10)]

    return xtr_list, ytr_list, xtrte_list, ytrte_list, xte_list, yte_list#, xte, yte
