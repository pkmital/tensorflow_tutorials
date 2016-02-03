"""In progress.

Parag K. Mital, Jan 2016
"""
import tensorflow as tf
import math
import numpy as np
from utils import corrupt, weight_variable, bias_variable


# %%
def VAE(input_shape=[None, 784],
        n_components_encoder=200,
        n_components_decoder=200,
        n_hidden=20,
        continuous=False,
        denoising=False,
        debug=False):
    # %%
    # Input placeholder
    if debug:
        input_shape = [50, 784]
        x = tf.Variable(np.zeros((input_shape), dtype=np.float32))
    else:
        x = tf.placeholder(tf.float32, input_shape)

    print('* Input')
    print('X:', x.get_shape().as_list())

    # %%
    # Optionally apply noise
    if denoising:
        print('* Denoising')
        x_noise = corrupt(x)
    else:
        x_noise = x

    if continuous:
        activation = lambda x: tf.log(1 + tf.exp(x))
    else:
        activation = lambda x: tf.tanh(x)

    dims = x_noise.get_shape().as_list()
    n_features = dims[1]

    print('* Encoder')
    W_enc = weight_variable([n_features, n_components_encoder])
    b_enc = bias_variable([n_components_encoder])
    h_enc = activation(tf.matmul(x_noise, W_enc) + b_enc)
    print('in:', x_noise.get_shape().as_list(),
          'W_enc:', W_enc.get_shape().as_list(),
          'b_enc:', b_enc.get_shape().as_list(),
          'h_enc:', h_enc.get_shape().as_list())

    print('* Variational Autoencoder')
    W_mu = weight_variable([n_components_encoder, n_hidden])
    b_mu = bias_variable([n_hidden])

    W_log_sigma = weight_variable([n_components_encoder, n_hidden])
    b_log_sigma = bias_variable([n_hidden])

    z_mu = tf.matmul(h_enc, W_mu) + b_mu
    z_log_sigma = 0.5 * (tf.matmul(h_enc, W_log_sigma) + b_log_sigma)
    print('in:', h_enc.get_shape().as_list(),
          'W_mu:', W_mu.get_shape().as_list(),
          'b_mu:', b_mu.get_shape().as_list(),
          'z_mu:', z_mu.get_shape().as_list())
    print('in:', h_enc.get_shape().as_list(),
          'W_log_sigma:', W_log_sigma.get_shape().as_list(),
          'b_log_sigma:', b_log_sigma.get_shape().as_list(),
          'z_log_sigma:', z_log_sigma.get_shape().as_list())
    # %%
    # Sample from noise distribution p(eps) ~ N(0, 1)
    if debug:
        epsilon = tf.random_normal(
            [dims[0], n_hidden])
    else:
        epsilon = tf.random_normal(
            tf.pack([tf.shape(x)[0], n_hidden]))
    print('epsilon:', epsilon.get_shape().as_list())

    # Sample from posterior
    z = z_mu + tf.exp(z_log_sigma) * epsilon
    print('z:', z.get_shape().as_list())

    print('* Decoder')
    W_dec = weight_variable([n_hidden, n_components_decoder])
    b_dec = bias_variable([n_components_decoder])
    h_dec = activation(tf.matmul(z, W_dec) + b_dec)
    print('in:', z.get_shape().as_list(),
          'W_dec:', W_dec.get_shape().as_list(),
          'b_dec:', b_dec.get_shape().as_list(),
          'h_dec:', h_dec.get_shape().as_list())

    W_mu_dec = weight_variable([n_components_decoder, n_features])
    b_mu_dec = bias_variable([n_features])
    y = tf.nn.sigmoid(tf.matmul(h_dec, W_mu_dec) + b_mu_dec)
    print('in:', z.get_shape().as_list(),
          'W_mu_dec:', W_mu_dec.get_shape().as_list(),
          'b_mu_dec:', b_mu_dec.get_shape().as_list(),
          'y:', y.get_shape().as_list())

    W_log_sigma_dec = weight_variable([n_components_decoder, n_features])
    b_log_sigma_dec = bias_variable([n_features])
    y_log_sigma = 0.5 * (
        tf.matmul(h_dec, W_log_sigma_dec) + b_log_sigma_dec)
    print('in:', z.get_shape().as_list(),
          'W_log_sigma_dec:', W_log_sigma_dec.get_shape().as_list(),
          'b_log_sigma_dec:', b_log_sigma_dec.get_shape().as_list(),
          'y_log_sigma:', y_log_sigma.get_shape().as_list())

    # p(x|z)
    if continuous:
        log_px_given_z = tf.reduce_sum(
            -(0.5 * tf.log(2.0 * np.pi) + y_log_sigma) -
            0.5 * tf.square((x - y) / tf.exp(y_log_sigma)))
    else:
        log_px_given_z = tf.reduce_sum(
            x * tf.log(y) +
            (1 - x) * tf.log(1 - y))

    # d_kl(q(z|x)||p(z))
    # Appendix B: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = 0.5 * tf.reduce_sum(
        1.0 + 2.0 * z_log_sigma - tf.square(z_mu) - tf.exp(2.0 * z_log_sigma))

    print('* Output')
    print('Y:', y.get_shape().as_list())

    loss = -(log_px_given_z + kl_div)

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
    learning_rate = 0.002
    optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # %%
    # Fit all training data
    batch_size = 100
    n_epochs = 50
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train = np.array([(img - mean_img) for img in batch_xs])
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
            np.reshape(test_xs[example_i, :], (28, 28)),
            cmap='gray')
        axs[1][example_i].imshow(
            np.reshape(
                np.reshape(recon[example_i, ...], (784,)) + mean_img,
                (28, 28)),
            cmap='gray')
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()
