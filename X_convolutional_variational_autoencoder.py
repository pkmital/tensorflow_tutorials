"""Convolutional variational autoencoder.

Parag K. Mital, Jan 2016
"""
import tensorflow as tf
import numpy as np
from utils import weight_variable, bias_variable, montage
from time import sleep
import matplotlib.pyplot as plt


# %%
def VAE(input_shape=[None, 784],
        n_filters=[784, 512],
        filter_sizes=[],
        n_hidden=256,
        n_code=2,
        activation=tf.nn.relu,
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
    # ensure 2-d is converted to square tensor.
    if convolutional:
        if len(x.get_shape()) == 2:
            x_dim = np.sqrt(x.get_shape().as_list()[1])
            if x_dim != int(x_dim):
                raise ValueError('Unsupported input dimensions')
            x_dim = int(x_dim)
            x_tensor = tf.reshape(
                x, [-1, x_dim, x_dim, 1])
        elif len(x.get_shape()) == 4:
            x_tensor = x
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
    encoder = []
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
            encoder.append(W)
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
        flattened = tf.reshape(current_input, [dims[0], size]
            if debug else tf.pack([tf.shape(x)[0], size]))

    else:
        size = dims[1]
        flattened = current_input

    print('* Reshape')
    print(current_input.get_shape().as_list(),
          '->', flattened.get_shape().as_list())

    print('* FC Layer')
    W_fc = weight_variable([size, n_hidden])
    b_fc = bias_variable([n_hidden])
    h = activation(tf.matmul(flattened, W_fc) + b_fc)

    print('in:', current_input.get_shape().as_list(),
          'W_fc:', W_fc.get_shape().as_list(),
          'b_fc:', b_fc.get_shape().as_list(),
          'h:', h.get_shape().as_list())

    print('* Variational Autoencoder')
    W_mu = weight_variable([n_hidden, n_code])
    b_mu = bias_variable([n_code])

    W_sigma = weight_variable([n_hidden, n_code])
    b_sigma = bias_variable([n_code])

    z_mu = tf.matmul(h, W_mu) + b_mu
    z_log_sigma = 0.5 * tf.matmul(h, W_sigma) + b_sigma

    print('in:', h.get_shape().as_list(),
          'W_mu:', W_mu.get_shape().as_list(),
          'b_mu:', b_mu.get_shape().as_list(),
          'mu:', z_mu.get_shape().as_list())
    print('in:', h.get_shape().as_list(),
          'W_sigma:', W_sigma.get_shape().as_list(),
          'b_sigma:', b_sigma.get_shape().as_list(),
          'log_sigma:', z_log_sigma.get_shape().as_list())
    # %%
    # Sample from noise distribution p(eps) ~ N(0, 1)
    if debug:
        epsilon = tf.random_normal(
            [dims[0], n_code])
    else:
        epsilon = tf.random_normal(
            tf.pack([tf.shape(x)[0], n_code]))

    # Sample from posterior
    z = z_mu + tf.mul(epsilon, tf.exp(z_log_sigma))

    print('z:', z.get_shape().as_list())

    print('* Decoder')

    W_dec = weight_variable([n_code, n_hidden])
    b_dec = bias_variable([n_hidden])
    h_dec = activation(tf.matmul(z, W_dec) + b_dec)

    print('in:', z.get_shape().as_list(),
          'W_dec:', W_dec.get_shape().as_list(),
          'b_dec:', b_dec.get_shape().as_list(),
          'h_dec:', h_dec.get_shape().as_list())

    W_fc_t = weight_variable([n_hidden, size])
    b_fc_t = bias_variable([size])
    h_fc_dec = activation(tf.matmul(h_dec, W_fc_t) + b_fc_t)

    print('in:', h_dec.get_shape().as_list(),
          'W_fc_t:', W_fc_t.get_shape().as_list(),
          'b_fc_t:', b_fc_t.get_shape().as_list(),
          'h_fc_dec:', h_fc_dec.get_shape().as_list())
    if convolutional:
        h_tensor = tf.reshape(
            h_fc_dec, [dims[0], dims[1], dims[2], dims[3]] if debug
            else tf.pack([tf.shape(x)[0], dims[1], dims[2], dims[3]]))
    else:
        h_tensor = h_fc_dec

    encoder.reverse()
    shapes.reverse()
    n_filters.reverse()

    print('* Reshape')
    print(h_fc_dec.get_shape().as_list(),
          '->', h_tensor.get_shape().as_list())
    # %%
    # Decoding layers
    current_input = h_tensor
    for layer_i, n_output in enumerate(n_filters[:-1][::-1]):
        n_input = n_filters[layer_i]
        n_output = n_filters[layer_i + 1]

        print(n_input, n_output)
        shape = shapes[layer_i]
        if convolutional:
            b = bias_variable([n_output])
            output = activation(tf.add(
                tf.nn.deconv2d(
                    current_input, encoder[layer_i],
                    shape if debug else
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

    dec_flat = tf.reshape(
        current_input, tf.pack([tf.shape(x)[0], input_shape[1]]))

    # %%
    # An extra fc layer and nonlinearity
    W_fc_final = weight_variable([input_shape[1], input_shape[1]])
    b_fc_final = bias_variable([input_shape[1]])
    y = tf.nn.sigmoid(tf.matmul(dec_flat, W_fc_final) + b_fc_final)

    # p(x|z)
    log_px_given_z = -tf.reduce_sum(
        x * tf.log(y + 1e-10) +
        (1 - x) * tf.log(1 - y + 1e-10), 1)

    # d_kl(q(z|x)||p(z))
    # Appendix B: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * tf.reduce_sum(
        1.0 + 2.0 * z_log_sigma - tf.square(z_mu) - tf.exp(2.0 * z_log_sigma),
        1)
    loss = tf.reduce_mean(log_px_given_z + kl_div)

    return {'cost': loss, 'x': x, 'z': z, 'y': y, 'filters': encoder}


def plot_reconstructions(xs, ys, fig=None, axs=None):
    # %%
    # Plot example reconstructions
    if fig is None or axs is None:
        fig, axs = plt.subplots(2, len(ys), figsize=(10, 2))
    for example_i in range(len(ys)):
        axs[0][example_i].imshow(
            np.reshape(xs[example_i, :], (28, 28)),
            cmap='gray')
        axs[1][example_i].imshow(
            np.reshape(
                np.reshape(ys[example_i, ...], (784,)),
                (28, 28)),
            cmap='gray')
    fig.show()
    fig.canvas.draw()
    fig.canvas.flush_events()


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
    ae = VAE()

    # %%
    learning_rate = 0.02
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # %%
    # Fit all training data
    batch_size = 100
    n_epochs = 10
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for epoch_i in range(n_epochs):
        print('--- Epoch', epoch_i)
        train_cost = 0
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train_cost += sess.run([ae['cost'], optimizer],
                                   feed_dict={ae['x']: batch_xs})[0]

            if batch_i % 5 == 0:
                xs, ys = mnist.test.images, mnist.test.labels
                zs = sess.run(ae['z'], feed_dict={ae['x']: xs})
                plt.cla()
                ax.scatter(zs[:, 0], zs[:, 1], c=np.argmax(ys, 1), alpha=0.5)
                ax.set_xlim([-3, 3])
                ax.set_ylim([-3, 3])
                fig.show()
                fig.canvas.draw()
                fig.canvas.flush_events()

        print('Train cost:', train_cost /
              (mnist.train.num_examples // batch_size))

        valid_cost = 0
        for batch_i in range(mnist.validation.num_examples // batch_size):
            batch_xs, _ = mnist.validation.next_batch(batch_size)
            valid_cost += sess.run([ae['cost']],
                                   feed_dict={ae['x']: batch_xs})[0]
        print('Validation cost:', valid_cost /
              (mnist.validation.num_examples // batch_size))

    

if __name__ == '__main__':
    test_mnist()
