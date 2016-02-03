"""Loading datasets.

Parag K. Mital, Jan. 2016
"""
import tensorflow.examples.tutorials.mnist.input_data as input_data


def MNIST(one_hot=True):
    """Returns the MNIST dataset.

    Returns
    -------
    mnist : DataSet
        DataSet object w/ convenienve props for accessing
        train/validation/test sets and batches.
    """
    return input_data.read_data_sets('MNIST_data/', one_hot=one_hot)
