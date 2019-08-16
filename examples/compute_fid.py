import os

import numpy as np

from transfer_gan.datasets import DatasetLoader
from transfer_gan.models._base_gan import _FrechetInceptionDistance


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    n_fid_sample = 10000
    inception_input_shape = (96, 96, 3)

    loader = DatasetLoader()
    (x_train, y_train), (x_test, y_test), class_names = loader.load_tiny_imagenet_subset()

    # Image scale sigmoid
    idx = np.random.randint(0, x_train.shape[0], n_fid_sample)
    x_tr_batch_1 = x_train[idx]
    x_tr_batch_1 = x_tr_batch_1 / 255.0

    idx = np.random.randint(0, x_train.shape[0], n_fid_sample)
    x_tr_batch_2 = x_train[idx]
    x_tr_batch_2 = x_tr_batch_2 / 255.0

    fid_sigmoid = _FrechetInceptionDistance(inception_input_shape=inception_input_shape)
    fid_sigmoid.compute_real_image_mean_and_cov(x_tr_batch_1)
    print("FID(sigmoid): %.2f" % fid_sigmoid.compute_fid(x_tr_batch_2))

    # Image scale tanh
    x_tr_batch_1 = (x_tr_batch_1 * 2.) - 1.
    x_tr_batch_2 = (x_tr_batch_2 * 2.) - 1.

    fid_tanh = _FrechetInceptionDistance(inception_input_shape=inception_input_shape)
    fid_tanh.compute_real_image_mean_and_cov(x_tr_batch_1)
    print("FID(tanh): %.2f" % fid_tanh.compute_fid(x_tr_batch_2))
