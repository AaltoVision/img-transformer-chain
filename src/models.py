from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import ops
import functools
import tensorflow as tf
import tensorflow.contrib.slim as slim


conv    = functools.partial(slim.conv2d, activation_fn=None)
deconv  = functools.partial(slim.conv2d_transpose, activation_fn=None)
relu    = tf.nn.relu
lrelu   = functools.partial(ops.leak_relu, leak=0.2)


def discriminator(img, scope, df_dim=64, reuse=False, train=True):
    bn  = functools.partial(slim.batch_norm, scale=True, is_training=train,
                            decay=0.9, epsilon=1e-5, updates_collections=None)

    with tf.variable_scope(scope + '_discriminator', reuse=reuse):
        h0 = lrelu(   conv(img, df_dim * 1, 4, 2, scope='h0_conv'))    # h0 is (128 x 128 x df_dim)
        h1 = lrelu(bn(conv(h0,  df_dim * 2, 4, 2, scope='h1_conv'), scope='h1_bn'))  # h1 is (64 x 64 x df_dim*2)
        h2 = lrelu(bn(conv(h1,  df_dim * 4, 4, 2, scope='h2_conv'), scope='h2_bn'))  # h2 is (32x 32 x df_dim*4)
        h3 = lrelu(bn(conv(h2,  df_dim * 8, 4, 1, scope='h3_conv'), scope='h3_bn'))  # h3 is (32 x 32 x df_dim*8)
        h4 = conv(h3, 1, 4, 1, scope='h4_conv')  # h4 is (32 x 32 x 1)

        return h4

def generator(img, scope, gf_dim=64, reuse=False, train=True):
    bn  = functools.partial(slim.batch_norm, scale=True, is_training=train,
                            decay=0.9, epsilon=1e-5, updates_collections=None)

    def res_block(x, dim, scope='res'):
        y = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        y = relu(bn(conv(y, dim, 3, 1, padding='VALID', scope=scope + '_conv1'), scope=scope + '_bn1'))
        y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        y = bn(conv(y, dim, 3, 1, padding='VALID', scope=scope + '_conv2'), scope=scope + '_bn2')
        return y + x

    with tf.variable_scope(scope + '_generator', reuse=reuse):
        c = tf.pad(img, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c = relu(bn(conv(c, gf_dim * 1, 7, 1, scope='c1_conv', padding='VALID'), scope='c1_bn'))
        c = relu(bn(conv(c, gf_dim * 2, 3, 2, scope='c2_conv'), scope='c2_bn'))
        c = relu(bn(conv(c, gf_dim * 4, 3, 2, scope='c3_conv'), scope='c3_bn'))

        r = c
        for i in range(9):
            r = res_block(r, gf_dim * 4, scope='r{}'.format(i+1))

        d = relu(bn(deconv(r, gf_dim * 2, 3, 2, scope='d1_dconv'), scope='d1_bn'))
        d = relu(bn(deconv(d, gf_dim, 3, 2, scope='d2_dconv'), scope='d2_bn'))
        d = tf.pad(d, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = conv(d, 3, 7, 1, padding='VALID', scope='pred_conv')
        pred = tf.nn.tanh(pred)

        return pred
