import pickle
import numpy as np
import pdb
from paths import MNIST_PATH
from tensorflow.examples.tutorials.mnist import input_data

def one_hot_maker(y, n_c=1623):
    n = len(y) if type(y) is list else 1
    one_hot = np.zeros((n, n_c))
    one_hot[range(n), y] = 1

    return one_hot

def subset_maker(x, y, y_onehot, nlist, n_c):
    x_list = [x[y==k][:nlist[k],:] for k in range(n_c)]
    y_list = [y[y==k][:nlist[k]] for k in range(n_c)]
    y_onehot_list = [y_onehot[y==k][:nlist[k],:] for k in range(n_c)]

    return x_list, y_list, y_onehot_list

omni_path = '/st1/dhna/data/omniglot/train_val_test_split.pkl'

def omni_input(nlist, n_tr = 15):
    with open(omni_path, 'rb') as file:
        splits = pickle.load(file)

    x = np.concatenate((splits[0], splits[2], splits[4]), axis=0)
    y = np.concatenate((splits[1], splits[3], splits[5]), axis=0)

    inst_idx = [np.where(y==k)[0] for k in range(len(nlist))]
    tr_x = [x[inst_idx[k][:nlist[k]]] for k in range(len(nlist))]
    tr_y = [[k]*nlist[k] for k in range(len(nlist))]
    te_x = [x[inst_idx[k][n_tr:n_tr+4]] for k in range(len(nlist))]
    te_y = [[k]*(4) for k in range(len(inst_idx))]
    te_nlist = [4 for k in range(len(inst_idx))]
    #te_y = [[k]*(len(inst_idx[k])-n_tr) for k in range(len(inst_idx))]
    #te_nlist = [(len(inst_idx[k])-n_tr) for k in range(len(inst_idx))]
    tr_y = [one_hot_maker(tr_y[k]) for k in range(len(nlist))]
    te_y =[one_hot_maker(te_y[k]) for k in range(len(nlist))]

    te_x_concat = np.reshape(te_x, [-1, np.shape(te_x)[-1]])
    te_y_concat = np.reshape(te_y, [-1, np.shape(te_y)[-1]])

    tr_nlist = nlist

    return tr_x, tr_y, te_x, te_y, te_x_concat, te_y_concat, tr_nlist, te_nlist

def omnist_imbal_input(onlist, mnlist):
    tr_x, te_x = [], []
    #omniglot
    otr_csel = np.arange(0,1523)
    np.random.shuffle(otr_csel)
    otr_csel = otr_csel[:20]

    ote_csel = np.arange(1523,1623)
    np.random.shuffle(ote_csel)
    ote_csel = ote_csel[:20]

    with open(omni_path, 'rb') as file:
        # if python 2
        splits = pickle.load(file)
        # if python 3
        #splits = pickle.load(file, encoding='latin1')

    otr_x, oval_x, ote_x = splits[0], splits[2], splits[4]
    otr_y, oval_y, ote_y = splits[1], splits[3], splits[5]

    otr_x = np.concatenate((otr_x, oval_x))
    otr_y = np.concatenate((otr_y, oval_y))

    tr_x.append([otr_x[np.where(otr_y == otr_csel[k])[0][:onlist[k]]] for k in range(20)])
    te_x.append([ote_x[np.where(ote_y == ote_csel[k])[0][:onlist[k]]] for k in range(20)])

    #MNIST
    mnist = input_data.read_data_sets(MNIST_PATH, one_hot=True, validation_size=0)
    mxtr, mytr = mnist.train.images, mnist.train.labels

    mytr_ = np.argmax(mytr, 1)
    tr_x.append([mxtr[mytr_==k][:mnlist[k],:] for k in range(5)])
    te_x.append([mxtr[mytr_==k][:mnlist[k-5],:] for k in range(5,10)])

    np.squeeze(tr_x)
    np.squeeze(te_x)

    return np.concatenate((tr_x[0], tr_x[1])), np.concatenate((te_x[0], te_x[1])), np.concatenate((onlist, mnlist)), np.concatenate((onlist, mnlist))

def omni_original_input():
    with open(omni_path, 'rb') as file:
        # if python 2
        splits = pickle.load(file)
        # if python 3
        #splits = pickle.load(file, encoding='latin1')

    tr_x, val_x, te_x = splits[0], splits[2], splits[4]
    tr_y, val_y, te_y = splits[1], splits[3], splits[5]

    tr_nlist = [len(np.where(tr_y == k)[0]) for k in range(1200)]
    val_nlist = [len(np.where(val_y == k)[0]) for k in range(1200,1623-100)]
    te_nlist = [len(np.where(te_y == k)[0]) for k in range(1623-100,1623)]

    tr_x = [tr_x[np.where(tr_y == k)[0]] for k in range(1200)]
    val_x = [val_x[np.where(val_y == k)[0]] for k in range(1200,1623-100)]
    te_x = [te_x[np.where(te_y == k)[0]] for k in range(1623-100,1623)]

    return tr_x, val_x, te_x, tr_y, val_y, te_y, tr_nlist, val_nlist, te_nlist

def omni_mcc_input():
    with open(omni_path, 'rb') as file:
        # if python 2
        splits = pickle.load(file)
        # if python 3
        #splits = pickle.load(file, encoding='latin1')

    tr_x, val_x, te_x = splits[0], splits[2], splits[4]
    tr_y, val_y, te_y = splits[1], splits[3], splits[5]

    tr_x = [tr_x[np.where(tr_y == k)[0]] for k in range(1200)]
    val_x = [val_x[np.where(val_y == k)[0]] for k in range(1200,1623-100)]
    te_x = [te_x[np.where(te_y == k)[0]] for k in range(1623-100,1623)]

    tmp = np.concatenate((tr_x, val_x, te_x))
    mcc_tr_x = [tmp[k][:10] for k in range(len(tmp))]
    mcc_te_x = [tmp[k][10:] for k in range(len(tmp))]

    tr_nlist = [len(mcc_tr_x[k]) for k in range(1623)]
    te_nlist = [len(mcc_te_x[k]) for k in range(1623)]
    return mcc_tr_x, mcc_te_x, tr_nlist, te_nlist


def omni_rotated_input():
    with open(omni_path, 'rb') as file:
        # if python 2
        splits = pickle.load(file)

    tr_x, val_x, te_x = splits[0], splits[2], splits[4]
    tr_y, val_y, te_y = splits[1], splits[3], splits[5]

    tr = np.load('/st1/dhna/anyshot_paper/data/rotated-omniglot/train.npy')
    val = np.load('/st1/dhna/anyshot_paper/data/rotated-omniglot/val.npy')
    te = np.load('/st1/dhna/anyshot_paper/data/rotated-omniglot/test.npy')

    print ("tr  shape ", np.shape(tr))
    print ("val shape ", np.shape(val))
    print ("te  shape ", np.shape(te))

    train, validation, test = [], [], []
    for k in range(1200):
        for j in range(len(tr)):
            train.append(np.reshape(tr[j][np.where(tr_y == k)[0]], (-1, 784)))
    for k in range(1200,1623-100):
        for j in range(len(val)):
            validation.append(np.reshape(val[j][np.where(val_y == k)[0]], (-1, 784)))
    for k in range(1623-100,1623):
        for j in range(len(tr)):
            test.append(np.reshape(te[j][np.where(te_y == k)[0]], (-1, 784)))

    train_nlist = [len(train[k]) for k in range(len(train))]
    validation_nlist = [len(validation[k]) for k in range(len(validation))]
    test_nlist = [len(test[k]) for k in range(len(test))]

    return train, validation, test, train_nlist, validation_nlist, test_nlist

def omni_rotated_mcc_input():
    with open(omni_path, 'rb') as file:
        # if python 2
        splits = pickle.load(file)

    tr_x, val_x, te_x = splits[0], splits[2], splits[4]
    tr_y, val_y, te_y = splits[1], splits[3], splits[5]

    tr = np.load('/st1/dhna/anyshot_paper/data/rotated-omniglot/train.npy')
    val = np.load('/st1/dhna/anyshot_paper/data/rotated-omniglot/val.npy')
    te = np.load('/st1/dhna/anyshot_paper/data/rotated-omniglot/test.npy')

    print ("tr  shape ", np.shape(tr))
    print ("val shape ", np.shape(val))
    print ("te  shape ", np.shape(te))

    train, validation, test = [], [], []
    for k in range(1200):
        for j in range(len(tr)):
            train.append(np.reshape(tr[j][np.where(tr_y == k)[0]], (-1, 784)))
    for k in range(1200,1623-100):
        for j in range(len(val)):
            validation.append(np.reshape(val[j][np.where(val_y == k)[0]], (-1, 784)))
    for k in range(1623-100,1623):
        for j in range(len(tr)):
            test.append(np.reshape(te[j][np.where(te_y == k)[0]], (-1, 784)))

    train_nlist = [len(train[k]) for k in range(len(train))]
    validation_nlist = [len(validation[k]) for k in range(len(validation))]
    test_nlist = [len(test[k]) for k in range(len(test))]

    tmp = np.concatenate([train, validation, test])
    mcc_tr_x = [tmp[k][:10] for k in range(len(tmp))]
    mcc_te_x = [tmp[k][10:] for k in range(len(tmp))]
    tr_nlist = [len(mcc_tr_x[k]) for k in range(1623)]
    te_nlist = [len(mcc_te_x[k]) for k in range(1623)]

    return mcc_tr_x, mcc_te_x, tr_nlist, te_nlist

def omnist_imbal_input2(onlist, mnlist):
    tr_x, trte_x, te_x = [],[],[]
    #omniglot
    otr_csel = np.arange(0,1523,1)
    otr_csel = otr_csel[:len(onlist)]

    ote_csel = np.arange(1523,1623,1)
    ote_csel = ote_csel[:len(onlist)]

    with open(omni_path, 'rb') as file:
        splits = pickle.load(file)

    otr_x, oval_x, ote_x = splits[0], splits[2], splits[4]
    otr_y, oval_y, ote_y = splits[1], splits[3], splits[5]

    otr_x = np.concatenate((otr_x, oval_x))
    otr_y = np.concatenate((otr_y, oval_y))

    tr_x.append([otr_x[np.where(otr_y == otr_csel[k])[0][:onlist[k]]] for k in range(len(onlist))])
    trte_x.append([otr_x[np.where(otr_y == otr_csel[k])[0][onlist[k]:onlist[k]*2]] for k in range(len(onlist))])
    te_x.append([ote_x[np.where(ote_y == ote_csel[k])[0][:onlist[k]]] for k in range(len(onlist))])

    #MNIST
    mnist = input_data.read_data_sets(MNIST_PATH, one_hot=True, validation_size=0)
    mxtr, mytr = mnist.train.images, mnist.train.labels

    mytr_ = np.argmax(mytr, 1)
    tr_x.append([mxtr[mytr_==k][:mnlist[k],:] for k in range(5)])
    trte_x.append([mxtr[mytr_==k][mnlist[k]:mnlist[k]*2,:] for k in range(5)])
    te_x.append([mxtr[mytr_==k][:mnlist[k-5],:] for k in range(5,10)])

    np.squeeze(tr_x)
    np.squeeze(trte_x)
    np.squeeze(te_x)
    tmp_trx, tmp_trtex, tmp_tex = [],[],[]
    for k in range(len(onlist)):
        tmp_trx.append(tr_x[0][k])
        tmp_trtex.append(trte_x[0][k])
        tmp_tex.append(te_x[0][k])

    for k in range(len(mnlist)):
        tmp_trx.append(tr_x[1][k])
        tmp_trtex.append(trte_x[1][k])
        tmp_tex.append(te_x[1][k])

    tr_x = tmp_trx
    trte_x = tmp_trtex
    te_x = tmp_tex
    nlist = np.concatenate((onlist, mnlist))

    return tr_x, trte_x, te_x, nlist, nlist, nlist

def omnist_small_imbal_val(onlist, mnlist):
    tr_x, val_x, te_x = [],[],[]
    #omniglot
    otr_csel = np.arange(0,1523,1)

    ote_csel = np.arange(1523,1623,1)

    with open(omni_path, 'rb') as file:
        splits = pickle.load(file)

    otr_x, oval_x, ote_x = splits[0], splits[2], splits[4]
    otr_y, oval_y, ote_y = splits[1], splits[3], splits[5]

    otr_x = np.concatenate((otr_x, oval_x))
    otr_y = np.concatenate((otr_y, oval_y))

    nxtr = [len(np.where(otr_y == otr_csel[k])[0]) for k in range(len(otr_csel))]
    nxte = [len(np.where(ote_y == ote_csel[k])[0]) for k in range(len(ote_csel))]

    cnt = 0
    k = 0
    while len(tr_x) < len(onlist[0][0]):
        if nxtr[cnt] == 20:
            tr_x.append(otr_x[np.where(otr_y == otr_csel[cnt])[0][:onlist[0][0][k]]])
            k = k + 1
        cnt = cnt + 1

    k = 0
    while len(val_x) < len(onlist[1][0]):
        if nxtr[cnt] == 20:
            val_x.append(otr_x[np.where(otr_y == otr_csel[cnt])[0][:onlist[1][0][k]]])
            k = k + 1
        cnt = cnt + 1

    cnt = 0
    k =0
    while len(te_x) < len(onlist[2][0]):
        if nxte[cnt] == 20:
            te_x.append(ote_x[np.where(ote_y == ote_csel[cnt])[0][:onlist[2][0][k]]])
            k = k + 1
        cnt = cnt + 1

    #MNIST
    mnist = input_data.read_data_sets(MNIST_PATH, one_hot=True, validation_size=0)
    mxtr, mytr = mnist.train.images, mnist.train.labels

    mytr_ = np.argmax(mytr, 1)
    for k in [0,1,2,3,4]:
        tr_x.append(mxtr[mytr_==k][:mnlist[0][0][k],:])
    for k in [5,6]:
        val_x.append(mxtr[mytr_==k][:mnlist[1][0][k-5],:])
    for k in [7,8,9]:
        te_x.append(mxtr[mytr_==k][:mnlist[2][0][k-7],:])

    nlist = [np.concatenate((onlist[k][0], mnlist[k][0]), 0) for k in range(3)]

    return tr_x, val_x, te_x, nlist[0], nlist[1], nlist[2]

def omnist_imbal_input(onlist, mnlist):
    tr_x, trte_x, te_x = [],[],[]
    #omniglot
    otr_csel = np.arange(0,1523,1)
    otr_csel = otr_csel[:len(onlist)]

    ote_csel = np.arange(1523,1623,1)
    ote_csel = ote_csel[:len(onlist)]

    with open(omni_path, 'rb') as file:
        splits = pickle.load(file)

    otr_x, oval_x, ote_x = splits[0], splits[2], splits[4]
    otr_y, oval_y, ote_y = splits[1], splits[3], splits[5]

    otr_x = np.concatenate((otr_x, oval_x))
    otr_y = np.concatenate((otr_y, oval_y))

    tr_x.append([otr_x[np.where(otr_y == otr_csel[k])[0][:onlist[k]]] for k in range(len(onlist))])
    trte_x.append([otr_x[np.where(otr_y == otr_csel[k])[0][onlist[k]:onlist[k]*2]] for k in range(len(onlist))])
    te_x.append([ote_x[np.where(ote_y == ote_csel[k])[0][:onlist[k]]] for k in range(len(onlist))])

    #MNIST
    mnist = input_data.read_data_sets(MNIST_PATH, one_hot=True, validation_size=0)
    mxtr, mytr = mnist.train.images, mnist.train.labels

    mytr_ = np.argmax(mytr, 1)
    tr_x.append([mxtr[mytr_==k][:mnlist[k],:] for k in range(5)])
    trte_x.append([mxtr[mytr_==k][mnlist[k]:mnlist[k]*2,:] for k in range(5)])
    te_x.append([mxtr[mytr_==k][:mnlist[k-5],:] for k in range(5,10)])

    np.squeeze(tr_x)
    np.squeeze(trte_x)
    np.squeeze(te_x)

    tr_x = np.concatenate([tr_x[0], tr_x[1]])
    trte_x = np.concatenate([trte_x[0], trte_x[1]])
    te_x = np.concatenate([te_x[0], te_x[1]])
    nlist = np.concatenate((onlist, mnlist))

    return tr_x, trte_x, te_x, nlist, nlist, nlist
