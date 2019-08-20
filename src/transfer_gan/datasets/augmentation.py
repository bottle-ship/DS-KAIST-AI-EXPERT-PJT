import numpy as np


def image_mixup(data, one_hot_labels, alpha=1):
    batch_size = len(data)
    weights = np.random.beta(alpha, alpha, batch_size)
    index = np.random.permutation(batch_size)
    x1, x2 = data, data[index]
    x = np.array([x1[i] * weights[i] + x2[i] * (1 - weights[i]) for i in range(len(weights))])
    y1 = np.array(one_hot_labels).astype(np.float)
    y2 = np.array(np.array(one_hot_labels)[index]).astype(np.float)
    y = np.array([y1[i] * weights[i] + y2[i] * (1 - weights[i]) for i in range(len(weights))])

    return x, y
