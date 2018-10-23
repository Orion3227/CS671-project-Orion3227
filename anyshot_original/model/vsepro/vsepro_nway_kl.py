import numpy as np
from ..layers import *
# from ..utils.train import *
from utils.train import *
dist = tf.contrib.distributions

print ("[INFO] Model file, ", __file__, " called.")

def encoder(x, cdim, zdim, training, name='encoder', reuse=None):
    keep_prob = 0.9 if training else 1.0
    x = dense(x, 500, activation=elu, name=name+'/dense1', reuse=reuse)
    x = dropout(x, keep_prob)
    x = dense(x, 500, activation=tanh, name=name+'/dense2', reuse=reuse)
    x = dropout(x, keep_prob)

    # q(c|X)
    vX = tf.expand_dims(tf.reduce_mean(x, 0), 0)
    qcX_mu = dense(vX, cdim, name=name+'/qcx_mu', reuse=reuse)
    qcX_sigma = dense(vX, cdim, activation=softplus,
            name=name+'/qcx_sigma', reuse=reuse)

    # q(c|x)
    qcx_mu = dense(x, cdim, name=name+'/qcx_mu', reuse=True)
    qcx_sigma = dense(x, cdim, activation=softplus,
            name=name+'/qcx_sigma', reuse=True)

    # q(z|x)
    qzx_mu = dense(x, zdim, name=name+'/qzx_mu', reuse=reuse)
    qzx_sigma = dense(x, zdim, activation=softplus,
            name=name+'/qzx_sigma', reuse=reuse)
    return qcX_mu, qcX_sigma, qcx_mu, qcx_sigma, qzx_mu, qzx_sigma

def decoder(x, training, name='decoder', reuse=None):
    keep_prob = 0.9 if training else 1.0
    x = dense(x, 500, activation=tanh, name=name+'/dense2', reuse=reuse)
    x = dropout(x, keep_prob)
    x = dense(x, 500, activation=elu, name=name+'/dense1', reuse=reuse)
    x = dropout(x, keep_prob)
    x = dense(x, 784, activation=sigmoid, name=name+'/output', reuse=reuse)
    return x

def autoencoder(args, x, cdim, zdim, training, name='autoencoder',
        reuse=None):
    qcX_mu, qcX_sigma, qcx_mu, qcx_sigma, qzx_mu, qzx_sigma = \
            encoder(x, cdim, zdim, training, reuse=reuse)

    qcX = Normal(qcX_mu, qcX_sigma)
    qzx = Normal(qzx_mu, qzx_sigma)

    cX = tf.tile(qcX.sample(), [tf.shape(x)[0], 1])
    zx = qzx.sample()
    cXzx = tf.concat([cX, zx], 1)
    qcX_sample = tf.squeeze(qcX.sample(100))

    # xll(decoder), kl, elbo
    x_hat = decoder(cXzx, training, reuse=reuse)
    xll = tf.reduce_sum(x*log(x_hat) + (1-x)*log(1-x_hat))

    c_kl = 0.5*tf.reduce_sum(qcX_mu**2+qcX_sigma**2-log(qcX_sigma**2)-1)
    z_kl = 0.5*tf.reduce_sum(qzx_mu**2+qzx_sigma**2-log(qzx_sigma**2)-1)
    elbo = xll - z_kl - args.gamma*c_kl

    aenet = {}
    aenet['elbo'] = elbo
    aenet['weights'] = tf.trainable_variables()

    aenet['qcX_mu'] = qcX_mu
    aenet['qcX_sigma'] = qcX_sigma
    aenet['qcX_sample'] = qcX_sample
    aenet['qzx_mu'] = qzx_mu
    aenet['qcx_mu'] = qcx_mu
    aenet['qcx_sigma'] = qcx_sigma
    aenet['qcx_sample'] = Normal(qcx_mu, qcx_sigma).sample([100])
    return aenet

def prototypical(args, qcX, cx, label=None):
    loss = 0.
    correct = 0.
    N = 0.
    for k in range(args.way):
        c_nont = []
        for j in range(args.way):
            logqcXj = tf.reduce_sum(dist.kl_divergence(qcX[j], cx[k]),1)
            c_nont.append(logqcXj)
        matrix = tf.stack(c_nont, 1)
        numer = -matrix[:,k]
        denom = log(tf.reduce_sum(exp(-matrix), 1))
        loss += tf.reduce_sum(-numer + denom)

        if label is not None:
            crt = tf.equal(tf.argmin(matrix, 1), tf.argmax(label[k], 1))
            correct += tf.reduce_sum(tf.cast(crt, tf.float32))
            N += tf.cast(tf.shape(label[k])[0], tf.float32)

    accuracy = correct / N if label is not None else 0.
    return loss, accuracy

def combine(args, x_emb, x_qry, y_qry, training, name='combine', reuse=None):

    # autoencoding module
    elbo = 0.
    qcx_mu, qzx_mu, qcX_mu, qcX_mu_tile, qcX_sample, qcx_sample = [], [], [], [], [], []
    qcX_sigma = []
    # prototypical module
    qcX, cx = [], []
    qry_mu, qry_sigma = [], []
    for k in range(args.way):
        #ELBO and support points
        aenet = autoencoder(args, x_emb[k], args.cdim, args.zdim, training,
                reuse=reuse if k == 0 else True)

        elbo += aenet['elbo']

        qcx_mu.append(aenet['qcx_mu'])
        qzx_mu.append(aenet['qzx_mu'])
        qcX_mu.append(aenet['qcX_mu'])
        qcX_mu_tile.append(tf.tile(aenet['qcX_mu'], [100, 1])) # 100*K
        qcX_sample.append(aenet['qcX_sample'])
        qcx_sample.append(aenet['qcx_sample'])

        qcX_sigma.append(aenet['qcX_sigma'])
        qcX.append(Normal(aenet['qcX_mu'], aenet['qcX_sigma']))

        # query points
        _, _, qry_qcx_mu, qry_qcx_sigma, _, _ = \
        encoder(x_qry[k], args.cdim, args.zdim, training, reuse=True)
        cx.append(Normal(qry_qcx_mu, qry_qcx_sigma))
        qry_mu.append(qry_qcx_mu)
        qry_sigma.append(qry_qcx_sigma)

    net = {}
    net['elbo'] = elbo
    net['weights'] = tf.trainable_variables()

    net['qcX_mu'] = tf.concat(qcX_mu, 0)
    net['qcX_sample'] = tf.concat(qcX_sample, 0)
    net['qcx_sample'] = tf.concat(qcx_sample, 0)
    net['qzx_mu'] = tf.concat(qzx_mu, 0)
    net['qcXqzx_mu'] = tf.concat([tf.concat(qcX_mu_tile, 0), net['qzx_mu']], 1)
    net['qcx_mu'] = tf.concat(qcx_mu, 0)
    net['qcxqzx_mu'] = tf.concat([net['qcx_mu'], net['qzx_mu']], 1)

    net['qcX_sigma'] = tf.concat(qcX_sigma, 0)
    net['qry_mu'] = tf.concat(qry_mu, 0)
    net['qry_sigma'] = tf.concat(qry_sigma, 0)

    net['proto_loss'], net['acc'] = prototypical(args, qcX, cx, label=y_qry)
    return net
