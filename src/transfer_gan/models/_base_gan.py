import os

import numpy as np
import pandas as pd
import tensorflow as tf

from abc import abstractmethod
from scipy import linalg
from skimage.transform import resize
from tensorflow.python.keras.utils import plot_model

from ..utils.keras_utils import load_model_from_json, save_model_to_json
from ..utils.os_utils import make_directory
from ..utils.pickle_utils import load_from_pickle, save_to_pickle

from ._base import BaseModel


#  https://stackoverflow.com/questions/55421153/fr%c3%a9chet-inception-distance-parameters-choice-in-tensorflow
def compute_fid(mean1, cov1, mean2, cov2):
    sum_sq_diff = np.sum((mean1 - mean2) ** 2)
    cov_mean = linalg.sqrtm(cov1.dot(cov2))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    fid = sum_sq_diff + np.trace(cov1 + cov2 - 2.0 * cov_mean)

    return fid


class BaseGAN(BaseModel):

    def __init__(self, input_shape,
                 noise_dim,
                 fake_activation='tanh',
                 optimizer='adam',
                 learning_rate=1e-4,
                 adam_beta_1=0.9,
                 adam_beta_2=0.999,
                 batch_size=64,
                 epochs=15,
                 n_fid_samples=5000,
                 **kwargs):
        super(BaseGAN, self).__init__()

        self.input_shape = input_shape
        self.noise_dim = noise_dim
        self.fake_activation = fake_activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_fid_samples = n_fid_samples

        self.input_channel_ = self.input_shape[-1]

        self._gene = self._build_generator()
        self._validate_generator_output_shape()
        self._disc = self._build_discriminator()

        self._gene_optimizer = self._set_optimizer()
        self._disc_optimizer = self._set_optimizer()

        self.inception_input_shape = None
        self.inception = None
        if self.n_fid_samples > 0:
            self._load_inception_model()

        self.history = list()

    def _initialize(self):
        self.input_channel_ = self.input_shape[-1]

        self._gene_optimizer = self._set_optimizer()
        self._disc_optimizer = self._set_optimizer()

        self.history.clear()

    def _scaling_image(self, x):
        if self.fake_activation == 'sigmoid':
            scaled_x = x / 255.0
        elif self.fake_activation == 'tanh':
            scaled_x = x / 255.0
            scaled_x = (scaled_x * 2.) - 1.
        else:
            supported_fake_activations = ('sigmoid', 'tanh')
            raise ValueError(
                "The fake activation '%s' is not supported. Supported activations are %s." %
                (self.fake_activation, supported_fake_activations)
            )

        return scaled_x

    def _unscaling_image(self, x):
        if self.fake_activation == 'tanh':
            x = x / 2 + 0.5
        x = x * 255.0

        return x

    @abstractmethod
    def _build_generator(self):
        raise NotImplementedError

    @abstractmethod
    def _build_discriminator(self):
        raise NotImplementedError

    def _validate_generator_output_shape(self):
        if not self.input_shape == self._gene.output_shape[1:]:
            raise ValueError(
                "Mismatch input shape(%s) and generator output shape(%s)" %
                (self.input_shape, self._gene.output_shape[1:])
            )

    def _set_optimizer(self):
        if self.optimizer.lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                lr=self.learning_rate, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2
            )
        else:
            raise ValueError("The optimizer '%s' is not supported." % self.optimizer)

        return optimizer

    def _load_inception_model(self):
        self.inception_input_shape = (96, 96, 3)
        self.inception = tf.contrib.keras.applications.InceptionV3(
            include_top=False, pooling='avg', input_shape=self.inception_input_shape
        )

    def _compute_image_mean_and_cov(self, images):
        images = np.asarray([resize(image, self.inception_input_shape, 0) for image in images])
        outputs = self.inception.predict(images)

        mean = outputs.mean(axis=0)
        cov = np.cov(outputs, rowvar=False)

        return mean, cov

    @abstractmethod
    def fit(self, x, y, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, **kwargs):
        raise NotImplementedError

    def show_generator_model(self, filename='generator.png'):
        self._gene.summary()
        plot_model(self._gene, filename)

    def show_discriminator_model(self, filename='discriminator.png'):
        self._disc.summary()
        plot_model(self._disc, filename)

    def save_model(self, model_dir_name):
        make_directory(model_dir_name)

        save_to_pickle(self.get_params(), os.path.join(model_dir_name, 'params.pkl'))

        save_model_to_json(self._gene, os.path.join(model_dir_name, 'generator_model.json'))
        self._gene.save_weights(os.path.join(model_dir_name, 'generator_weights.h5'))

        save_model_to_json(self._disc, os.path.join(model_dir_name, 'discriminator_model.json'))
        self._disc.save_weights(os.path.join(model_dir_name, 'discriminator_weights.h5'))

        pd.DataFrame(self.history, columns=['Epochs', 'Loss_Disc', 'Loss_Gene', 'FID']).to_pickle(
            os.path.join(model_dir_name, 'history.pkl')
        )

    def load_model(self, model_dir_name):
        self.set_params(**load_from_pickle(os.path.join(model_dir_name, 'params.pkl')))

        self._initialize()

        self._gene = load_model_from_json(os.path.join(model_dir_name, 'generator_model.json'))
        self._gene.load_weights(os.path.join(model_dir_name, 'generator_weights.h5'))

        self._disc = load_model_from_json(os.path.join(model_dir_name, 'discriminator_model.json'))
        self._disc.load_weights(os.path.join(model_dir_name, 'discriminator_weights.h5'))

        self.history = pd.read_pickle(os.path.join(model_dir_name, 'history.pkl')).values.tolist()
