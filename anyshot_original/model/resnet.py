import tensorflow as tf
from utils.train import *
relu = tf.nn.relu

def dense(x, units, name='dense', reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        with tf.device('/cpu:0'):
            kernel = tf.get_variable('kernel', [x.shape[1].value, units])
            bias = tf.get_variable('bias', [units],
                    initializer=tf.zeros_initializer())
            x = tf.matmul(x, kernel) + bias
            return x

def conv(x, filters, kernel_size=3, strides=1, padding='SAME',
        name='conv', reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        with tf.device('/cpu:0'):
            kernel = tf.get_variable('kernel',
                    [kernel_size, kernel_size, x.shape[1].value, filters])
        x = tf.nn.conv2d(x, kernel, [1, 1, strides, strides],
                padding=padding, data_format='NCHW')
        return x

def batch_norm(x, training, decay=0.9, name='batch_norm', reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        with tf.device('/cpu:0'):
            dim = x.shape[1].value
            moving_mean = tf.get_variable('moving_mean', [dim],
                    initializer=tf.zeros_initializer(), trainable=False)
            moving_var = tf.get_variable('moving_var', [dim],
                    initializer=tf.ones_initializer(), trainable=False)
            beta = tf.get_variable('beta', [dim],
                    initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma', [dim],
                    initializer=tf.ones_initializer())

        if training:
            x, batch_mean, batch_var = tf.nn.fused_batch_norm(x, gamma, beta, data_format='NCHW')
            update_mean = moving_mean.assign_sub((1-decay)*(moving_mean - batch_mean))
            update_var = moving_var.assign_sub((1-decay)*(moving_var - batch_var))
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_var)
        else:
            x, batch_mean, batch_var = tf.nn.fused_batch_norm(x, gamma, beta,
                    mean=moving_mean, variance=moving_var, is_training=False,
                    data_format='NCHW')
        return x

def global_avg_pool(x):
    return tf.reduce_mean(x, [2, 3])

def basic(x, filters, training, strides=1,
        name='block', reuse=None, dropout=None, **dropout_kwargs):
    shortcut = x if filters==x.shape[1].value else \
            conv(x, filters, kernel_size=1, strides=strides,
                    name=name+'/conv0', reuse=reuse)
    x = conv(relu(batch_norm(x, training, name=name+'/bn1', reuse=reuse)),
            filters, strides=strides, name=name+'/conv1', reuse=reuse)
    x = conv(relu(batch_norm(x, training, name=name+'/bn2', reuse=reuse)),
            filters, name=name+'/conv2', reuse=reuse)
    x = x if dropout is None else \
            dropout(x, training, name=name+'/dropout', reuse=reuse,
                    **dropout_kwargs)
    return shortcut + x

def bottleneck(x, filters, training, strides=1,
        name='block', reuse=None, dropout=None, **dropout_kwargs):
    shortcut = x if 4*filters==x.shape[1].value else \
            conv(x, 4*filters, kernel_size=1, strides=strides,
                    name=name+'/conv0', reuse=reuse)
    x = conv(relu(batch_norm(x, training, name=name+'/bn1', reuse=reuse)),
            filters, kernel_size=1, name=name+'/conv1', reuse=reuse)
    x = conv(relu(batch_norm(x, training, name=name+'/bn2', reuse=reuse)),
            filters, strides=strides, name=name+'/conv2', reuse=reuse)
    x = conv(relu(batch_norm(x, training, name=name+'/bn3', reuse=reuse)),
            4*filters, kernel_size=1, name=name+'/conv3', reuse=reuse)
    x = x if dropout is None else \
            dropout(x, training, name=name+'/dropout', reuse=reuse,
                    **dropout_kwargs)
    return shortcut + x

def group(x, filters, block, n_blocks, training,
        strides=1, name='group', reuse=None,
        dropout=None, **dropout_kwargs):
    for i in range(n_blocks):
        x = block(x, filters, training, strides=(strides if i==0 else 1),
                name=name+'/block'+str(i+1), reuse=reuse,
                dropout=dropout, **dropout_kwargs)
    return x

def resnet(x, y, filters, depth, n_classes, training,
        name='resnet', reuse=None, 
        decay=1e-4, dropout=None, **dropout_kwargs):
    if depth == 18:
        n_blocks = [2, 2, 2, 2]
        block = basic
    elif depth == 34:
        n_blocks = [3, 4, 6, 3]
        block = basic
    elif depth == 50:
        n_blocks = [3, 4, 6, 3]
        block = bottleneck
    elif depth == 101:
        n_blocks = [3, 4, 23, 3]
        block = bottleneck
    elif depth == 152:
        n_blocks = [3, 8, 36, 3]
        block = bottleneck
    else:
        raise ValueError('Invalid depth %d' % depth)

    x = conv(x, filters, name=name+'/pre_conv', reuse=reuse)
    for i, nb in enumerate(n_blocks):
        x = group(x, filters, block, nb, training, strides=(1 if i==0 else 2),
                name=name+'/group'+str(i+1), reuse=reuse,
                dropout=dropout, **dropout_kwargs)
        filters *= 2
    x = relu(batch_norm(x, training, name=name+'/post_bn', reuse=reuse))
    x = global_avg_pool(x)
    x = dense(x, n_classes, name=name+'/post_fc', reuse=reuse)

    net = {}
    all_vars = tf.get_collection('variables', scope=name)
    net['qpi_vars'] = [v for v in all_vars if 'qpi_vars' in v.name]
    net['pzx_vars'] = [v for v in all_vars if 'pzx_vars' in v.name]
    net['weights'] = [v for v in all_vars \
            if 'qpi_vars' not in v.name and 'pzx_vars' not in v.name]

    net['cent'] = cross_entropy(x, y)
    net['wd'] = weight_decay(decay, var_list=net['weights'])
    net['acc'] = accuracy(x, y)

    prefix = 'train_' if training else 'test_'
    net['kl'] = tf.get_collection('kl')
    net['pi'] = tf.get_collection(prefix+'pi')
    net['n_active'] = tf.get_collection(prefix+'n_active')

    return net
