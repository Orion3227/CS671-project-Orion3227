from layers import *
from utils.train import *

def lenet_dense(x, y, training, name='lenet', reuse=None):
    x = dense(x, 500, activation=relu, name=name+'/dense1', reuse=reuse)
    x = dense(x, 300, activation=relu, name=name+'/dense2', reuse=reuse)
    x = dense(x, 10, name=name+'/dense3', reuse=reuse)

    net = {}
    all_vars = tf.get_collection('variables', scope=name)
    net['weights'] = [v for v in all_vars]
    net['cent'] = cross_entropy(x, y)
    net['wd'] = weight_decay(1e-4, var_list=net['weights'])
    net['acc'] = accuracy(x, y)
    return net

def lenet_conv(x, y, training, name='lenet', reuse=None):
    x = tf.reshape(x, [-1, 1, 28, 28])
    x = relu(conv(x, 20, 5, name=name+'/conv1', reuse=reuse))
    x = pool(x, name=name+'/pool1')
    x = relu(conv(x, 50, 5, name=name+'/conv2', reuse=reuse))
    x = pool(x, name=name+'/pool2')
    x = flatten(x)
    x = dense(x, 500, activation=relu, name=name+'/dense1', reuse=reuse)
    x = dense(x, 10, name=name+'/dense2', reuse=reuse)

    net = {}
    all_vars = tf.get_collection('variables', scope=name)
    net['weights'] = [v for v in all_vars]
    net['cent'] = cross_entropy(x, y)
    net['wd'] = weight_decay(1e-4, var_list=net['weights'])
    net['acc'] = accuracy(x, y)
    return net

def lenet_conv_anyshot(x, y, N, training, name='lenet', reuse=None):
    x = tf.reshape(x, [-1, 1, 28, 28])
    x = relu(conv(x, 20, 5, name=name+'/conv1', reuse=reuse))
    x = pool(x, name=name+'/pool1')
    x = relu(conv(x, 50, 5, name=name+'/conv2', reuse=reuse))
    x = pool(x, name=name+'/pool2')
    x = flatten(x)
    x = dense(x, 500, activation=relu, name=name+'/dense1', reuse=reuse)

    dim = 50
    idx = [tf.shape(tf.squeeze(tf.where(tf.equal(y[:,k],
        tf.ones_like(y[:,k])))))[0] for k in range(10)]
    N = tf.expand_dims(tf.stack(idx, 0), 1)

    v = tf.divide(tf.matmul(tf.transpose(y), x), tf.cast(N, tf.float32))
           # tf.expand_dims(tf.convert_to_tensor(idx, dtype=tf.float32), 1))

    mu = dense(v, dim, name=name+'emb_mu', reuse=reuse)
    sigma = dense(v, dim, activation=softplus, name=name+'emb_sigma', reuse=reuse)
    emu = dense(x, dim, name=name+'emb_mu', reuse=True)
    #esigma = dense(x, dim, activation=softplus, name=name+'emb_sigma', reuse=reuse)

    pc = Normal(loc=tf.zeros([dim]), scale=tf.ones([dim]))
    qc = [Normal(loc=mu[k,:], scale=sigma[k,:]) for k in range(10)]
    qc_all = Normal(loc=mu, scale=sigma)

    spt = [qc_k.sample() for qc_k in qc]
    qry = [qc_k.sample() for qc_k in qc]
    spt_all = tf.stack(spt, 0)

    xll = tf.constant(0.)
    for k in range(10):
        x_k = tf.gather(x, idx[k], axis=0)

        #seed = tf.tile(tf.expand_dims(0.5*(spt[k] + qry[k]), 0), [idx[k], 1])
        #x_k_hat = dense(seed, 500, activation=relu,
        #        name=name+'/decoder_mu', reuse=reuse if k==0 else True)
        #xll -= tf.reduce_sum((x_k - x_k_hat)**2)

        seed = tf.expand_dims((spt[k] + qry[k])*0.5, 0)
        x_k_hat_mu = dense(seed, 500, activation=relu,
                name=name+'/decoder_mu', reuse=reuse if k==0 else True)
        x_k_hat_sigma = dense(seed, 500, activation=softplus,
                name=name+'/decoder_sigma', reuse=reuse if k==0 else True)
        x_k_hat = Normal(loc=x_k_hat_mu, scale=x_k_hat_sigma)
        xll += tf.reduce_sum(x_k_hat.log_prob(x_k))

    yll = tf.constant(0.)
    for k in range(10):
        lognumer = -tf.reduce_sum((spt[k] - qry[k])**2)
        logdenom = log(tf.reduce_sum(exp(-tf.reduce_sum(
            (spt_all - qry[k])**2, axis=1))))
        yll += lognumer - logdenom

    kl = tf.constant(0.)
    for k in range(10):
        kl += 2*tf.reduce_sum(kl_divergence(qc[k], pc))

    elbo = xll + yll - kl

    batchsize = 200
    ec_eval = [tf.reduce_sum(qc_all.log_prob(emu[i,:]), 1) for i in range(batchsize)]
    ec_eval = tf.stack(ec_eval, 0)

    net = {}
    all_vars = tf.get_collection('variables', scope=name)
    net['weights'] = [v for v in all_vars]
    net['cent'] = -elbo
    net['xll'] = -xll
    net['yll'] = -yll
    net['kl'] = kl
    net['wd'] = weight_decay(1e-4, var_list=net['weights'])
    net['acc'] = accuracy(ec_eval, y)
    return net

def lenet_conv_anyshot2(x, y, N, training, name='lenet', reuse=None):
    x = tf.reshape(x, [-1, 1, 28, 28])
    x = relu(conv(x, 20, 5, name=name+'/conv1', reuse=reuse))
    x = pool(x, name=name+'/pool1')
    x = relu(conv(x, 50, 5, name=name+'/conv2', reuse=reuse))
    x = pool(x, name=name+'/pool2')
    x = flatten(x)
    x = dense(x, 500, activation=relu, name=name+'/dense1', reuse=reuse)

    dim = 50
    idx = [tf.shape(tf.squeeze(tf.where(tf.equal(y[:,k],
        tf.ones_like(y[:,k])))))[0] for k in range(10)]
    N = tf.expand_dims(tf.stack(idx, 0), 1)

    v = tf.divide(tf.matmul(tf.transpose(y), x), tf.cast(N, tf.float32))
    ev = x

    mu = dense(v, 50, name=name+'/emb_mu', reuse=reuse)
    emu = dense(ev, 50, name=name+'/emb_mu', reuse=True)
    sigma = dense(v, 50, activation=softplus, name=name+'/emb_sigma', reuse=reuse)
    c = Normal(loc=mu, scale=sigma)
    c_support = c.sample()
    c_query = c.sample()

    loss = 0.
    for k in range(10):
        numer = -tf.reduce_sum((c_support[k,:] - c_query[k,:])**2)
        denom = log(tf.reduce_sum(exp(-tf.reduce_sum(
            (c_support - c_query[k,:])**2, axis=1))))
        loss += -numer + denom

    bs = 200
    ec_eval = tf.stack([tf.reduce_sum(c.prob(emu[i,:]), 1) for i in range(bs)], 0)

    net = {}
    all_vars = tf.get_collection('variables', scope=name)
    net['weights'] = [v for v in all_vars]
    net['cent'] = loss
    net['wd'] = weight_decay(1e-4, var_list=net['weights'])
    net['acc'] = accuracy(ec_eval, y)

    net['e_samp'] = emu
    net['c_samp'] = c.sample([30])
    net['c_mu'] = mu
    return net
