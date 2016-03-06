import os
import pickle
import numpy as np


def cifar10_download(dst='cifar10'):
    from six.moves import urllib
    import tarfile
    if not os.path.exists(dst):
        os.makedirs(dst)
    path = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filepath, _ = urllib.request.urlretrieve(path, './')
    tarfile.open(filepath, 'r:gz').extractall(dst)


def cifar10_load(dst='cifar10'):
    if not os.path.exists(dst):
        cifar10_download(dst)
    Xs = None
    ys = None
    for f in range(1, 6):
        cf = pickle.load(open(
            '%s/data_batch_%d' % (dst, f), 'rb'),
            encoding='LATIN')
        if Xs is not None:
            Xs = np.r_[Xs, cf['data']]
            ys = np.r_[ys, np.array(cf['labels'])]
        else:
            Xs = cf['data']
            ys = cf['labels']
    return Xs, ys


def dense_to_one_hot(labels, n_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    labels = np.array(labels)
    n_labels = labels.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot


class DatasetSplit(object):
    def __init__(self, images, labels):
        self.images = np.array(images).astype(np.float32)
        self.labels = np.array(labels).astype(np.int32)
        self.n_labels = len(np.unique(labels))
        self.num_examples = len(self.images)

    def next_batch(self, batch_size=100):
        # Shuffle each epoch
        current_permutation = np.random.permutation(range(len(self.images)))
        epoch_images = self.images[current_permutation, ...]
        epoch_labels = dense_to_one_hot(
            self.labels[current_permutation, ...], self.n_labels)

        # Then iterate over the epoch
        self.current_batch_idx = 0
        while self.current_batch_idx < len(self.images):
            end_idx = min(
                self.current_batch_idx + batch_size, len(self.images))
            this_batch = {
                'images': epoch_images[self.current_batch_idx:end_idx],
                'labels': epoch_labels[self.current_batch_idx:end_idx]
            }
            self.current_batch_idx += batch_size
            yield this_batch['images'], this_batch['labels']


class Dataset(object):
    def __init__(self, Xs, ys, split=[0.8, 0.1, 0.1]):

        self.all_idxs = []
        self.all_labels = []
        self.all_inputs = []
        self.train_idxs = []
        self.valid_idxs = []
        self.test_idxs = []
        self.n_labels = 0
        self.split = split

        # Now mix all the labels that are currently stored as blocks
        self.all_inputs = Xs
        self.all_labels = ys
        n_idxs = len(self.all_inputs)
        idxs = range(n_idxs)
        rand_idxs = np.random.permutation(idxs)
        self.all_inputs = self.all_inputs[rand_idxs, ...]
        self.all_labels = self.all_labels[rand_idxs, ...]

        # Get splits
        self.train_idxs = idxs[:round(split[0] * n_idxs)]
        self.valid_idxs = idxs[len(self.train_idxs):
                               len(self.train_idxs) + round(split[1] * n_idxs)]
        self.test_idxs = idxs[len(self.valid_idxs):
                              len(self.valid_idxs) + round(split[2] * n_idxs)]

    @property
    def train(self):
        inputs = self.all_inputs[self.train_idxs, ...]
        labels = self.all_labels[self.train_idxs, ...]
        return DatasetSplit(inputs, labels)

    @property
    def valid(self):
        inputs = self.all_inputs[self.valid_idxs, ...]
        labels = self.all_labels[self.valid_idxs, ...]
        return DatasetSplit(inputs, labels)

    @property
    def test(self):
        inputs = self.all_inputs[self.test_idxs, ...]
        labels = self.all_labels[self.test_idxs, ...]
        return DatasetSplit(inputs, labels)

    def mean(self):
        return np.mean(self.all_inputs, axis=0)

    def std(self):
        return np.std(self.all_inputs, axis=0)
