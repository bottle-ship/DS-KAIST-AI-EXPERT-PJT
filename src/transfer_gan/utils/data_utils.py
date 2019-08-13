import numpy as np


def get_data_information(x, y):
    input_shape = x.shape[1:]
    num_classes = len(np.unique(y))

    return input_shape, num_classes
