import os

import cv2
import numpy as np
import keras

from keras.utils import data_utils

from ..utils.pickle_utils import load_from_pickle
from ..utils.pickle_utils import save_to_pickle


class DatasetLoader(object):

    def __init__(self):
        self.datasets = keras.datasets
        self.get_file = data_utils.get_file

    def load_fashion_mnist(self):
        (x_train, y_train), (x_test, y_test) = self.datasets.fashion_mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        return (x_train, y_train), (x_test, y_test), class_names

    def load_cifar10(self):
        (x_train, y_train), (x_test, y_test) = self.datasets.cifar10.load_data()

        class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
                       'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

        return (x_train, y_train), (x_test, y_test), class_names

    def load_tiny_imagenet(self):
        dirname = 'tiny-imagenet-200.zip'
        origin = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        path = self.get_file(dirname, origin=origin, extract=True, archive_format='zip')
        path = path[:-4]

        if not os.path.exists(os.path.join(path, 'tiny-imagenet-x-train.npy')):
            with open(os.path.join(path, 'wnids.txt'), 'r') as f:
                wnids = [x.strip() for x in f]

            wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

            with open(os.path.join(path, 'words.txt'), 'r') as f:
                wnid_to_words = dict()
                for line in f:
                    line = line.split('\t')
                    wnid_to_words[line[0]] = line[1].replace('\n', '').split(',')
            class_names = [wnid_to_words[wnid] for wnid in wnids]
            save_to_pickle(class_names, os.path.join(path, 'tiny-imagenet-class-name.pkl'))

            x_train = list()
            y_train = list()
            for label in wnid_to_label.keys():
                target_dir = os.path.join(path, 'train', label, 'images')
                for filename in os.listdir(target_dir):
                    img = cv2.imread(os.path.join(target_dir, filename))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_AREA)
                    x_train.append(img)
                    y_train.append(wnid_to_label[label])

            x_train = np.array(x_train)
            y_train = np.array(y_train)

            np.save(os.path.join(path, 'tiny-imagenet-x-train.npy'), x_train)
            np.save(os.path.join(path, 'tiny-imagenet-y-train.npy'), y_train)

            with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
                val_annortations = dict()
                for line in f:
                    line = line.split('\t')
                    val_annortations[line[0]] = line[1]

            x_test = list()
            y_test = list()
            target_dir = os.path.join(path, 'val', 'images')
            for filename in val_annortations.keys():
                img = cv2.imread(os.path.join(target_dir, filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_AREA)
                x_test.append(img)
                y_test.append(wnid_to_label[val_annortations[filename]])

            x_test = np.array(x_test)
            y_test = np.array(y_test)

            np.save(os.path.join(path, 'tiny-imagenet-x-test.npy'), x_test)
            np.save(os.path.join(path, 'tiny-imagenet-y-test.npy'), y_test)

        else:
            x_train = np.load(os.path.join(path, 'tiny-imagenet-x-train.npy'))
            y_train = np.load(os.path.join(path, 'tiny-imagenet-y-train.npy'))
            x_test = np.load(os.path.join(path, 'tiny-imagenet-x-test.npy'))
            y_test = np.load(os.path.join(path, 'tiny-imagenet-y-test.npy'))
            class_names = load_from_pickle(os.path.join(path, 'tiny-imagenet-class-name.pkl'))

        return (x_train, y_train), (x_test, y_test), class_names


if __name__ == '__main__':
    loader = DatasetLoader()
    loader.load_cifar10()
