from transfer_gan.datasets import DatasetLoader
from transfer_gan.metrics.fid import create_realdata_stats


if __name__ == '__main__':
    loader = DatasetLoader()
    (x_train, y_train), (_, _), class_names = loader.load_cifar10()

    create_realdata_stats(x_train, './fid_stats_cifar10.npz')
