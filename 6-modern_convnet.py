"""Tutorial on how to build a convnet w/ modern changes, e.g.
Batch Normalization, Leaky rectifiers, and strided convolution.

Parag K. Mital, Jan 2016.
"""
# %%
import tensorflow as tf
from batch_norm import batch_norm
from activations import lrelu
from connections import conv2d, linear
from datasets import MNIST


# %% Setup input to the network and true output label.  These are
# simply placeholders which we'll fill in later.
mnist = MNIST()
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
x_tensor = tf.reshape(x, [-1, 28, 28, 1])

# %% Define the network:
bn1 = batch_norm(-1, name='bn1')
bn2 = batch_norm(-1, name='bn2')
bn3 = batch_norm(-1, name='bn3')
h_1 = lrelu(bn1(conv2d(x_tensor, 32, name='conv1')), name='lrelu1')
h_2 = lrelu(bn2(conv2d(h_1, 64, name='conv2')), name='lrelu2')
h_3 = lrelu(bn3(conv2d(h_2, 64, name='conv3')), name='lrelu3')
h_3_flat = tf.reshape(h_3, [-1, 64 * 4 * 4])
h_4 = linear(h_3_flat, 10)
y_pred = tf.nn.softmax(h_4)

# %% Define loss/eval/training functions
cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# %% We now create a new session to actually perform the initialization the
# variables:
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# %% We'll train in minibatches and report accuracy:
n_epochs = 10
batch_size = 100
for epoch_i in range(n_epochs):
    for batch_i in range(mnist.train.num_examples // batch_size):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={
            x: batch_xs, y: batch_ys})
    print(sess.run(accuracy,
                   feed_dict={
                       x: mnist.validation.images,
                       y: mnist.validation.labels
                   }))
