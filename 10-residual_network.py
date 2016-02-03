"""Tutorial on how to build a residual network.

Parag K. Mital, Jan 2016.
"""
# %%
import tensorflow as tf
from connections import conv2d, linear
from collections import namedtuple


# %%
def residual_network(input_shape, n_outputs,
                     activation=tf.nn.relu, debug=False):
    """Builds a residual network.

    Parameters
    ----------
    input_shape : list
        Input dimensions of tensor
    n_outputs : TYPE
        Number of outputs of final softmax
    activation : Attribute, optional
        Nonlinearity to apply after each convolution

    Returns
    -------
    name : TYPE
        Description
    """
    # %%
    LayerBlock = namedtuple(
        'LayerBlock', ['num_layers', 'num_filters', 'bottleneck_size'])
    blocks = [LayerBlock(3, 128, 32),
              LayerBlock(3, 256, 64),
              LayerBlock(3, 512, 128),
              LayerBlock(3, 1024, 256)]

    # %%
    x = tf.placeholder(tf.float32, input_shape, 'x')

    # %%
    # First convolution expands to 64 channels and downsamples
    net = conv2d(x, 64, k_h=7, k_w=7,
                 batch_norm=True, name='conv1',
                 activation=activation)

    # %%
    # Max pool and downsampling
    net = tf.nn.max_pool(
        net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # %%
    # Setup first chain of resnets
    net = conv2d(net, blocks[0].num_filters, k_h=1, k_w=1,
                 stride_h=1, stride_w=1, padding='VALID', name='conv2')

    # %%
    # Loop through all res blocks
    for block_i, block in enumerate(blocks):
        for layer_i in range(block.num_layers):

            name = 'block_%d/layer_%d' % (block_i, layer_i)
            conv = conv2d(net, block.num_filters, k_h=1, k_w=1,
                          padding='VALID', stride_h=1, stride_w=1,
                          activation=activation, batch_norm=True,
                          name=name + '/conv_in')

            conv = conv2d(conv, block.bottleneck_size, k_h=3, k_w=3,
                          padding='SAME', stride_h=1, stride_w=1,
                          activation=activation, batch_norm=True,
                          name=name + '/conv_bottleneck')

            conv = conv2d(conv, block.num_filters, k_h=1, k_w=1,
                          padding='VALID', stride_h=1, stride_w=1,
                          activation=activation, batch_norm=True,
                          name=name + '/conv_out')

            net = conv + net
        try:
            # upscale to the next block size
            next_block = blocks[block_i + 1]
            net = conv2d(net, next_block.num_filters, k_h=1, k_w=1,
                         padding='SAME', stride_h=1, stride_w=1, bias=False,
                         name='block_%d/conv_upscale' % block_i)
        except IndexError:
            pass

    # %%
    net = tf.reshape(
        tf.reduce_mean(net, 3),
        [-1, net.get_shape().as_list()[1] * net.get_shape().as_list()[2]])

    net = linear(net, n_outputs, activation=tf.nn.softmax)

    # %%
    return net
