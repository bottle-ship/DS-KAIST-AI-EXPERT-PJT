import numpy as np

from keras.preprocessing.image import ImageDataGenerator


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


def image_augmentation(x, y, batch_size=290, end_cnt=100, **kwargs):
    datagen = ImageDataGenerator(**kwargs)
    datagen.fit(x)

    x_sub = None
    y_sub = None

    gen_cnt = 0
    for x_batch, y_batch in datagen.flow(x, y, batch_size=batch_size):
        if x_sub is None and y_sub is None:
            x_sub = x_batch
            y_sub = y_batch
        else:
            x_sub = np.vstack((x_sub, x_batch))
            y_sub = np.hstack((y_sub, y_batch))

        gen_cnt += 1
        if gen_cnt > end_cnt:
            break

    x = np.vstack((x, x_sub))
    y = np.hstack((y, y_sub))

    return x, y
