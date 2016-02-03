"""Batch Normalization for TensorFlow.
Parag K. Mital, Jan 2016."""

import tensorflow as tf


class batch_norm(object):
    """Basic usage from: http://stackoverflow.com/a/33950177

    Parag K. Mital, Jan 2016

    Attributes
    ----------
    batch_size : int
        Size of the batch.  Set to -1 to fit to current net.
    beta : Tensor
        A 1D beta Tensor with size matching the last dimension of t.
        An offset to be added to the normalized tensor.
    ema : tf.train.ExponentialMovingAverage
        For computing the moving average.
    epsilon : float
        A small float number to avoid dividing by 0.
    gamma : Tensor
        If "scale_after_normalization" is true, this tensor will be multiplied
        with the normalized tensor.
    momentum : float
        The decay to use for the moving average.
    name : str
        The variable scope for all variables under batch normalization.
    """

    def __init__(self, batch_size, epsilon=1e-5,
                 momentum=0.1, name="batch_norm"):
        """Summary

        Parameters
        ----------
        batch_size : int
            Size of the batch, or -1 for size to fit.
        epsilon : float, optional
            A small float number to avoid dividing by 0.
        momentum : float, optional
            Decay to use for the moving average.
        name : str, optional
            Variable scope will be under this prefix.
        """
        with tf.variable_scope(name) as scope:
            self.epsilon = epsilon
            self.momentum = momentum
            self.batch_size = batch_size
            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        """Applies/updates the BN to the input Tensor.

        Parameters
        ----------
        x : Tensor
            The input tensor to normalize.
        train : bool, optional
            Whether or not to train parameters.

        Returns
        -------
        x_normed : Tensor
            The normalized Tensor.
        """
        shape = x.get_shape().as_list()

        # Using a variable scope means any new variables
        # will be prefixed with "variable_scope/", e.g.:
        # "variable_scope/new_variable".  Also, using
        # TensorBoard, this will make everything very
        # nicely grouped.
        with tf.variable_scope(self.name) as scope:
            self.gamma = tf.get_variable(
                "gamma", [shape[-1]],
                initializer=tf.random_normal_initializer(1., 0.02))
            self.beta = tf.get_variable(
                "beta", [shape[-1]],
                initializer=tf.constant_initializer(0.))

            mean, variance = tf.nn.moments(x, [0, 1, 2])

            return tf.nn.batch_norm_with_global_normalization(
                x, mean, variance, self.beta, self.gamma, self.epsilon,
                scale_after_normalization=True)
