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

def lenet_conv_anyshot2(x, y, N, training, name='lenet', reuse=None):
    x = tf.reshape(x, [-1, 1, 28, 28])
    x = relu(conv(x, 20, 5, name=name+'/conv1', reuse=reuse))
    x = pool(x, name=name+'/pool1')
    x = relu(conv(x, 50, 5, name=name+'/conv2', reuse=reuse))
    x = pool(x, name=name+'/pool2')
    x = flatten(x)
    x = dense(x, 500, activation=relu, name=name+'/dense1', reuse=reuse)

    if training:
        v = tf.matmul(tf.transpose(y),x)
    else:
        #v = v_all # get v from outside
        v = tf.matmul(tf.transpose(y),x)
    ev = x

    mu = dense(v, 50, name=name+'/emb_mu', reuse=reuse)
    sigma = dense(v, 50, activation=softplus, name=name+'/emb_sigma', reuse=reuse)
    c = Normal(loc=mu, scale=sigma)
    c_support = c.sample()
    c_query = c.sample()

    loss = 0.
    for k in range(10): # query
        numer = -tf.reduce_sum((c_support[k,:] - c_query[k,:])**2)
        denom = log(tf.reduce_sum(exp(-tf.reduce_sum(
            (c_support - c_query[k,:])**2, axis=1))))
        loss += -numer + denom

    emu = dense(ev, 50, name=name+'/emb_mu', reuse=True)
    esigma = dense(ev, 50, activation=softplus, name=name+'/emb_sigma',
            reuse=True)

    """
    kl = tf.constant(0.)

    if training:
        for k in range(10):
            y_k = y[:,k]
            idx_k = tf.squeeze(tf.where(tf.equal(y_k,tf.ones_like(y_k))))
            emu_k = tf.gather(emu, idx_k, axis=0)
            esigma_k = tf.gather(esigma, idx_k, axis=0)
            ec_k = Normal(loc=emu_k, scale=esigma_k)
            ec_k_samp = ec_k.sample()
            c_k = Normal(loc=mu[k,:], scale=sigma[k,:])

            for i in range(N[k]):
                kl += log(tf.reduce_mean(tf.reduce_prod(
                    ec_k.prob(ec_k_samp[i,:]), 1), 0)) \
                        - tf.reduce_sum(log(c_k.prob(ec_k_samp[i,:])))
    """

    """
    kl = tf.constant(0.)
    # mixture distributions
    for k in range(10):
        y_k = y[:,k]
        idx_k = tf.squeeze(tf.where(tf.equal(y_k,tf.ones_like(y_k))))
        emu_k = tf.gather(emu, idx_k, axis=0)
        esigma_k = tf.gather(esigma, idx_k, axis=0)

        mix_k = Mixture(
          cat=Categorical(probs=[1/float(N[k]) for _ in range(N[k])]),
          components=[Normal(loc=emu_k[i,:], scale=esigma_k[i,:])
              for i in range(N[k])])

        loc_k = mix_k.sample()
        mix_k_eval = mix_k.prob(loc_k)
        c_k_eval = Normal(loc=mu[k,:], scale=sigma[k,:]).prob(loc_k)

        kl += tf.reduce_sum(log(mix_k_eval) - log(c_k_eval))
    """

    #c_eval = []
    #for k in range(10):
    #    c_eval.append(tf.reduce_prod(c.prob(mu[k,:]), 1))
    #c_eval = tf.stack(c_eval, 0)

    batchsize = 1000
    ec_eval = []
    for i in range(batchsize):
        ec_eval.append(tf.reduce_sum(c.prob(emu[i,:]), 1))
        #ec_eval.append(-tf.reduce_sum((mu -
        #    tf.tile(tf.expand_dims(emu[i,:],0),[10,1]))**2, 1))
    ec_eval = tf.stack(ec_eval, 0)

    net = {}
    all_vars = tf.get_collection('variables', scope=name)
    net['weights'] = [v for v in all_vars]
    net['cent'] = loss #+ 1e-5*kl
    net['wd'] = weight_decay(1e-4, var_list=net['weights'])
    #net['acc'] = accuracy(c_eval, tf.eye(10))
    net['acc'] = accuracy(ec_eval, y)
    #net['idx'] = idx_k

    net['c_samp'] = c.sample([30])
    net['e_samp'] = emu
    return net

"""
def lenet_conv_anyshot_nouse(x, y, training, v_all=None, name='lenet', reuse=None):
    x = tf.reshape(x, [-1, 1, 28, 28])
    x = relu(conv(x, 20, 5, name=name+'/conv1', reuse=reuse))
    x = pool(x, name=name+'/pool1')
    x = relu(conv(x, 50, 5, name=name+'/conv2', reuse=reuse))
    x = pool(x, name=name+'/pool2')
    x = flatten(x)
    x = dense(x, 500, activation=relu, name=name+'/dense1', reuse=reuse)

    if training:
        v = tf.matmul(tf.transpose(y),x)
    else:
        #v = v_all # get v from outside
        v = tf.matmul(tf.transpose(y),x)
    ev = x

    #pi_h = tf.squeeze(dense(v, 1, name=name+'/emb_pi', reuse=reuse))
    mu = dense(v, 50, name=name+'/emb_mu', reuse=reuse)
    sigma = dense(v, 50, activation=softplus, name=name+'/emb_sigma',
            reuse=reuse)
    c = [Normal(loc=mu[k,:], scale=1e-1*sigma[k,:]) for k in range(10)]

    c_eval = []
    for k in range(10):
        c_nont = []
        c_k_samp = c[k].sample()
        for j in range(10):
            c_nont.append(tf.reduce_prod(c[j].prob(c_k_samp)))
        c_eval.append(tf.stack(c_nont))
    c_eval = tf.stack(c_eval, 0)

    c_numer = log(tf.reduce_sum(tf.multiply(c_eval, tf.eye(10)), 1))
    c_denom = log(tf.reduce_sum(c_eval, 1))
    cent = tf.reduce_sum(-c_numer + c_denom)

    #epi_h = tf.squeeze(dense(ev, 1, name=name+'/emb_pi', reuse=True))
    emu = dense(ev, 50, name=name+'/emb_mu', reuse=True)
    esigma = dense(ev, 50, activation=softplus, name=name+'/emb_sigma',
            reuse=True)

    kl = tf.constant(0.)
    for k in range(10):
        y_k = y[:,k]
        idx_k = tf.squeeze(tf.where(tf.equal(y_k,tf.ones_like(y_k))))
        epi_h_k = tf.gather(epi_h, idx_k, axis=0)
        emu_k = tf.gather(emu, idx_k, axis=0)
        esigma_k = tf.gather(esigma, idx_k, axis=0)

        ec_k_cat = RelaxedCategorical(temperature=0.01, logits=epi_h_k).sample()
        pec_k = Normal(loc=emu_k, scale=esigma_k)
        pec_k_loc = tf.reduce_sum(pec_k.sample() * tf.expand_dims(ec_k_cat, 1), 0)
        pec_k_eval = pec_k.prob(pec_k_loc)

        pc_k = Normal(loc=mu[k,:], scale=sigma[k,:])
        pc_k_eval = pc_k.prob(pec_k_loc)

        kl += tf.reduce_sum(log(pec_k_eval) - log(pc_k_eval))

    batchsize = 100
    ec_eval = []
    for i in range(batchsize):
        ec_nont = []
        for j in range(10):
            ec_nont.append(tf.reduce_prod(c[j].prob(emu[i,:])))
        ec_eval.append(tf.stack(ec_nont))
    ec_eval = tf.stack(ec_eval, 0)

    c = Normal(loc=mu, scale=sigma)
    c_eval = []
    for k in range(10):
        c_eval.append(tf.reduce_prod(c.prob(mu[k,:]), 1))
    c_eval = tf.stack(c_eval, 0)

    net = {}
    all_vars = tf.get_collection('variables', scope=name)
    net['weights'] = [v for v in all_vars]
    net['cent'] = 1e+4*cent# + kl/100000.
    net['wd'] = weight_decay(1e-4, var_list=net['weights'])
    net['acc'] = accuracy(c_eval, tf.eye(10))

    c = Normal(loc=mu, scale=sigma)
    net['c_samp'] = c.sample([30]) # type desired number of samples
    return net
"""
