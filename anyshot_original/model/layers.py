import tensorflow as tf
import numpy as np

exp = tf.exp
log = lambda x: tf.log(x + 1e-20)
logit = lambda x: log(x) - log(1-x)
softplus = tf.nn.softplus
softmax = tf.nn.softmax
tanh = tf.nn.tanh
relu = tf.nn.relu
elu = tf.nn.elu
sigmoid = tf.nn.sigmoid
dropout = tf.nn.dropout

dense = tf.layers.dense
flatten = tf.contrib.layers.flatten

def conv(x, filters, kernel_size=3, strides=1, **kwargs):
    return tf.layers.conv2d(x, filters, kernel_size, strides,
            data_format='channels_first', **kwargs)

def pool(x, **kwargs):
    return tf.layers.max_pooling2d(x, 2, 2,
            data_format='channels_first', **kwargs)

def global_avg_pool(x):
    return tf.reduce_mean(x, axis=[2, 3])

batch_norm = tf.layers.batch_normalization
layer_norm = tf.contrib.layers.layer_norm

# distributions
Normal = tf.distributions.Normal
Uniform = tf.distributions.Uniform
RelaxedCategorical = tf.contrib.distributions.RelaxedOneHotCategorical
Categorical = tf.contrib.distributions.Categorical
Mixture = tf.contrib.distributions.Mixture

# kl divergences
def kl_diagnormal_stdnormal(mu, sigma):
    a = mu**2
    b = sigma**2
    c = -1
    d = -log(sigma**2)
    return 0.5 * tf.reduce_sum(a + b + c + d)

def kl_diagnormal_diagnormal(q_mu, q_sigma, p_mu, p_sigma):
    a = log(p_sigma**2)
    b = -1
    c = -log(q_sigma**2)
    d = ((q_mu - p_mu)**2 + q_sigma**2) / (p_sigma**2)
    return 0.5 * tf.reduce_sum(a + b + c + d)
