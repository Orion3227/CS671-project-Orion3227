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

    dim = 200
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

def lenet(x, name, reuse):
    x = tf.reshape(x, [-1, 1, 28, 28])
    x = relu(conv(x, 20, 5, name=name+'/conv1', reuse=reuse))
    x = pool(x, name=name+'/pool1')
    x = relu(conv(x, 50, 5, name=name+'/conv2', reuse=reuse))
    x = pool(x, name=name+'/pool2')
    x = flatten(x)
    x = dense(x, 500, activation=relu, name=name+'/dense1', reuse=reuse)
    return x

def lenet_anyshot(x_s, y_s, x_q, y_q, xtrall, ytrall,
        training, dim=200, name='lenet', reuse=None):

    xtrall = lenet(xtrall, name, reuse)
    x_s = lenet(x_s, name, True)
    x_q = lenet(x_q, name, True)
    x = tf.concat([x_s, x_q], 0)
    y = tf.concat([y_s, y_q], 0)

    Ntrall = tf.expand_dims(tf.reduce_sum(ytrall,0), 1)
    vtrall = tf.divide(tf.matmul(tf.transpose(ytrall), xtrall), Ntrall)

    mu = dense(vtrall, dim, name=name+'/emb_mu', reuse=reuse)
    sigma = dense(vtrall, dim, activation=softplus, name=name+'/emb_sigma', reuse=reuse)
    c = Normal(loc=mu, scale=sigma)

    emu = dense(x, dim, name=name+'/emb_mu', reuse=True)

    if training:
        N_s = tf.expand_dims(tf.reduce_sum(y_s,0), 1)
        N_q = tf.expand_dims(tf.reduce_sum(y_q,0), 1)

        v_s = tf.divide(tf.matmul(tf.transpose(y_s), x_s), N_s)
        v_q = tf.divide(tf.matmul(tf.transpose(y_q), x_q), N_q)

        mu_s = dense(v_s, dim, name=name+'/emb_mu', reuse=True)
        sigma_s = dense(v_s, dim, activation=softplus, name=name+'/emb_sigma', reuse=True)
        mu_q = dense(v_q, dim, name=name+'/emb_mu', reuse=True)
        sigma_q = dense(v_q, dim, activation=softplus, name=name+'/emb_sigma', reuse=True)

        c_s = Normal(loc=mu_s, scale=sigma_s)
        c_q = Normal(loc=mu_q, scale=sigma_q)
        c_support = c_s.sample()
        c_query = c_q.sample()

        loss = 0.
        for k in range(10):
            numer = -tf.reduce_sum((c_support[k,:] - c_query[k,:])**2)
            denom = log(tf.reduce_sum(exp(-tf.reduce_sum(
                (c_support - c_query[k,:])**2, axis=1))))
            loss += -numer + denom

    ec_eval = []
    for k in range(10):
        c_k = Normal(loc=mu[k,:], scale=sigma[k,:])
        eval_k = tf.reduce_sum(c_k.log_prob(emu), 1)
        ec_eval.append(eval_k)
    ec_eval = tf.stack(ec_eval, 1)
    print(ec_eval)

    net = {}
    all_vars = tf.get_collection('variables', scope=name)
    net['weights'] = [v for v in all_vars]
    net['cent'] = loss if training else tf.constant(0.)
    net['wd'] = weight_decay(1e-4, var_list=net['weights'])
    net['acc'] = accuracy(ec_eval, y)

    net['e_samp'] = emu
    net['c_samp'] = c.sample([30])
    net['c_mu'] = mu
    return net
