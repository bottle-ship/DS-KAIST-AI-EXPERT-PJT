import numpy as np

from transfer_gan.datasets import DatasetLoader
from transfer_gan.metrics.fid import create_realdata_stats


if __name__ == '__main__':
    loader = DatasetLoader()
    (x_train, y_train), (_, _), class_names = loader.load_fashion_mnist()

    create_realdata_stats(np.repeat(x_train, 3, axis=3), './fid_stats_fashion_mnist.npz')
