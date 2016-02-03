"""In progress.

Parag K. Mital, Jan 2016
"""
import tensorflow as tf
import math
import numpy as np
from utils import corrupt, weight_variable, bias_variable


def normal2(x, mean, logvar):
    return (c -
            logvar / 2.0 -
            tf.pow((x - mean), 2.0) /
            (2.0 * tf.exp(logvar)))


# %%
def VAE(input_shape=[None, 784],
        n_filters=[],
        filter_sizes=[],
        n_hidden=512,
        n_code=64,
        activation=tf.nn.relu,
        denoising=False,
        convolutional=False,
        debug=False):
    # %%
    # Input placeholder
    if debug:
        input_shape = [50, 784]
        x = tf.Variable(np.zeros((input_shape), dtype=np.float32))
    else:
        x = tf.placeholder(tf.float32, input_shape)

    # %%
    # Optionally apply denoising autoencoder
    if denoising:
        x_noise = corrupt(x)
    else:
        x_noise = x

    # %%
    # ensure 2-d is converted to square tensor.
    if convolutional:
        if len(x.get_shape()) == 2:
            x_dim = np.sqrt(x_noise.get_shape().as_list()[1])
            if x_dim != int(x_dim):
                raise ValueError('Unsupported input dimensions')
            x_dim = int(x_dim)
            x_tensor = tf.reshape(
                x_noise, [-1, x_dim, x_dim, 1])
        elif len(x_noise.get_shape()) == 4:
            x_tensor = x_noise
        else:
            raise ValueError('Unsupported input dimensions')
    else:
        x_tensor = x
    current_input = x_tensor

    print('* Input')
    print('X:', current_input.get_shape().as_list())
    # %%
    # Build the encoder
    shapes = []
    print('* Encoder')
    for layer_i, n_input in enumerate(n_filters[:-1]):
        n_output = n_filters[layer_i + 1]
        shapes.append(current_input.get_shape().as_list())
        if convolutional:
            n_input = shapes[-1][3]
            W = weight_variable([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input, n_output])
            b = bias_variable([n_output])
            output = activation(
                tf.add(tf.nn.conv2d(
                    current_input, W,
                    strides=[1, 2, 2, 1], padding='SAME'), b))
        else:
            W = weight_variable([n_input, n_output])
            b = bias_variable([n_output])
            output = activation(tf.matmul(current_input, W) + b)
        print('in:', current_input.get_shape().as_list(),
              'W:', W.get_shape().as_list(),
              'b:', b.get_shape().as_list(),
              'out:', output.get_shape().as_list())
        current_input = output

    dims = current_input.get_shape().as_list()
    if convolutional:
        # %%
        # Flatten and build latent layer as means and standard deviations
        size = (dims[1] * dims[2] * dims[3])
        if debug:
            flattened = tf.reshape(current_input, [dims[0], size])
        else:
            flattened = tf.reshape(current_input,
                                   tf.pack([tf.shape(x)[0], size]))
    else:
        size = dims[1]
        flattened = current_input

    print('* Reshape')
    print(current_input.get_shape().as_list(),
          '->', flattened.get_shape().as_list())

    print('* FC Layer')
    W_fc = weight_variable([size, n_hidden])
    b_fc = bias_variable([n_hidden])
    h = tf.nn.tanh(tf.matmul(flattened, W_fc) + b_fc)
    print('in:', current_input.get_shape().as_list(),
          'W_fc:', W_fc.get_shape().as_list(),
          'b_fc:', b_fc.get_shape().as_list(),
          'h:', h.get_shape().as_list())

    print('* Variational Autoencoder')
    W_mu = weight_variable([n_hidden, n_code])
    b_mu = bias_variable([n_code])

    W_sigma = weight_variable([n_hidden, n_code])
    b_sigma = bias_variable([n_code])

    mu = tf.matmul(h, W_mu) + b_mu
    log_sigma = tf.mul(0.5, tf.matmul(h, W_sigma) + b_sigma)
    print('in:', h.get_shape().as_list(),
          'W_mu:', W_mu.get_shape().as_list(),
          'b_mu:', b_mu.get_shape().as_list(),
          'mu:', mu.get_shape().as_list())
    print('in:', h.get_shape().as_list(),
          'W_sigma:', W_sigma.get_shape().as_list(),
          'b_sigma:', b_sigma.get_shape().as_list(),
          'log_sigma:', log_sigma.get_shape().as_list())
    # %%
    # Sample from noise distribution p(eps) ~ N(0, 1)
    if debug:
        epsilon = tf.random_normal(
            [dims[0], n_code])
    else:
        epsilon = tf.random_normal(
            tf.pack([tf.shape(x)[0], n_code]))
    print('epsilon:', epsilon.get_shape().as_list())

    # Sample from posterior
    z = mu + tf.mul(epsilon, tf.exp(log_sigma))
    print('z:', z.get_shape().as_list())

    print('* Decoder')
    W_dec = weight_variable([n_code, n_hidden])
    b_dec = bias_variable([n_hidden])
    h_dec = tf.nn.relu(tf.matmul(z, W_dec) + b_dec)
    print('in:', z.get_shape().as_list(),
          'W_dec:', W_dec.get_shape().as_list(),
          'b_dec:', b_dec.get_shape().as_list(),
          'h_dec:', h_dec.get_shape().as_list())

    W_fc_t = weight_variable([n_hidden, size])
    b_fc_t = bias_variable([size])
    h_fc_dec = tf.nn.relu(tf.matmul(h_dec, W_fc_t) + b_fc_t)
    print('in:', h_dec.get_shape().as_list(),
          'W_fc_t:', W_fc_t.get_shape().as_list(),
          'b_fc_t:', b_fc_t.get_shape().as_list(),
          'h_fc_dec:', h_fc_dec.get_shape().as_list())

    if convolutional:
        if debug:
            h_tensor = tf.reshape(
                h_fc_dec, [dims[0], dims[1], dims[2], dims[3]])
        else:
            h_tensor = tf.reshape(
                h_fc_dec, tf.pack([tf.shape(x)[0], dims[1], dims[2], dims[3]]))
    else:
        h_tensor = h_fc_dec

    shapes.reverse()
    n_filters.reverse()

    print('* Reshape')
    print(h_fc_dec.get_shape().as_list(),
          '->', h_tensor.get_shape().as_list())

    ## %%
    ## Decoding layers
    current_input = h_tensor
    for layer_i, n_output in enumerate(n_filters[:-1][::-1]):
        n_input = n_filters[layer_i]
        n_output = n_filters[layer_i + 1]
        shape = shapes[layer_i]
        if convolutional:
            W = weight_variable([
                    filter_sizes[layer_i],
                    filter_sizes[layer_i],
                    n_output, n_input])
            b = bias_variable([n_output])
            if debug:
                output = activation(tf.add(
                    tf.nn.deconv2d(
                        current_input, W,
                        shape,
                        strides=[1, 2, 2, 1], padding='SAME'), b))
            else:
                output = activation(tf.add(
                    tf.nn.deconv2d(
                        current_input, W,
                        tf.pack(
                            [tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                        strides=[1, 2, 2, 1], padding='SAME'), b))
        else:
            W = weight_variable([n_input, n_output])
            b = bias_variable([n_output])
            output = activation(tf.matmul(current_input, W) + b)
        print('in:', current_input.get_shape().as_list(),
              'W:', W.get_shape().as_list(),
              'b:', b.get_shape().as_list(),
              'out:', output.get_shape().as_list())
        current_input = output

    # %%
    # Now have the reconstruction through the network
    y_tensor = current_input
    y = tf.reshape(y_tensor, tf.pack([tf.shape(x)[0], input_shape[1]]))

    print('* Output')
    print('Y:', y_tensor.get_shape().as_list())

    # %%
    # Log Prior: D_KL(q(z|x)||p(z))
    # Equation 10
    prior_loss = 0.5 * tf.reduce_sum(
        1.0 + 2.0 * log_sigma - tf.pow(mu, 2.0) - tf.exp(2.0 * log_sigma))

    # Reconstruction Cost
    recon_loss = tf.reduce_sum(tf.abs(y_tensor - x_tensor))

    # Total cost
    loss = recon_loss - prior_loss

    # log_px_given_z = normal2(x, mu, log_sigma)
    # loss = (log_pz + log_px_given_z - log_qz_given_x).sum()

    return {'cost': loss, 'x': x, 'z': z, 'y': y}


# %%
def test_mnist():
    """Summary

    Returns
    -------
    name : TYPE
        Description
    """
    # %%
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt

    # %%
    # load MNIST as before
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    ae = VAE()

    # %%
    learning_rate = 0.005
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # %%
    # Fit all training data
    batch_size = 100
    n_epochs = 100
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train = np.array([img - mean_img for img in batch_xs])
        print(epoch_i,
              sess.run([ae['cost'], optimizer],
                       feed_dict={ae['x']: train})[0])

    # %%
    # Plot example reconstructions
    n_examples = 12
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
    print(recon.shape)
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (28, 28)))
        axs[1][example_i].imshow(
            np.reshape(
                np.reshape(recon[example_i, ...], (784,)) + mean_img,
                (28, 28)))
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()
