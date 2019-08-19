import os


def _get_currnet_file_path():
    return os.path.dirname(os.path.abspath(__file__))


def get_fid_stats_path_fashion_mnist():
    path = os.path.join(_get_currnet_file_path(), '_fid_stats', 'fid_stats_fashion_mnist.npz')

    return path


def get_fid_stats_path_cifar10():
    path = os.path.join(_get_currnet_file_path(), '_fid_stats', 'fid_stats_cifar10.npz')

    return path


def get_fid_stats_path_tiny_imagenet():
    path = os.path.join(_get_currnet_file_path(), '_fid_stats', 'fid_stats_tiny_imagenet.npz')

    return path


def get_fid_stats_path_tiny_imagenet_subset():
    path = os.path.join(_get_currnet_file_path(), '_fid_stats', 'fid_stats_tiny_imagenet_subset.npz')

    return path
