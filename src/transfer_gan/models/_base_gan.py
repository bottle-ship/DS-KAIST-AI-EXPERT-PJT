import os

import numpy as np
import pandas as pd
import tensorflow as tf

from abc import abstractmethod
from scipy import linalg
from skimage.transform import resize
from tensorflow.python.keras import backend
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.utils import plot_model

from ..utils.keras_utils import load_model_from_json, save_model_to_json
from ..utils.os_utils import make_directory
from ..utils.pickle_utils import load_from_pickle, save_to_pickle

from ._base import BaseModel


class _FrechetInceptionDistance(object):
    #  https://stackoverflow.com/questions/55421153/fr%c3%a9chet-inception-distance-parameters-choice-in-tensorflow

    def __init__(self, inception_input_shape=(96, 96, 3),
                 batch_size=500):
        self.inception_input_shape = inception_input_shape
        self.batch_size = batch_size

        self._real_mean = None
        self._real_cov = None

    def _predict_inception(self, images):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        inception_sess = tf.compat.v1.Session(config=config)
        backend.set_session(inception_sess)

        images = np.asarray([resize(image, self.inception_input_shape, 0) for image in images])

        inception = InceptionV3(include_top=False, pooling='avg', input_shape=self.inception_input_shape)

        outputs = None
        while images.shape[0] > 0:
            images_batch = images[:self.batch_size]
            images = images[self.batch_size:]
            outputs_batch = inception.predict(images_batch)

            if outputs is None:
                outputs = outputs_batch
            else:
                outputs = np.vstack((outputs, outputs_batch))

        inception_sess.__del__()

        return outputs

    def initialize(self):
        self._real_mean = None
        self._real_cov = None

    def compute_real_image_mean_and_cov(self, real_images):
        outputs = self._predict_inception(real_images)
        self._real_mean = outputs.mean(axis=0)
        self._real_cov = np.cov(outputs, rowvar=False)

    def compute_fid(self, fake_images):
        if self._real_mean is not None and self._real_cov is not None:
            outputs = self._predict_inception(fake_images)
            fake_mean = outputs.mean(axis=0)
            fake_cov = np.cov(outputs, rowvar=False)

            sum_sq_diff = np.sum((self._real_mean - fake_mean) ** 2)
            cov_mean = linalg.sqrtm(self._real_cov.dot(fake_cov))
            if np.iscomplexobj(cov_mean):
                cov_mean = cov_mean.real
            fid = sum_sq_diff + np.trace(self._real_cov + fake_cov - 2.0 * cov_mean)

            return fid
        else:
            assert "You must run 'compute_real_image_mean_and_cov' method first."


class BaseGAN(BaseModel):

    def __init__(self, input_shape,
                 noise_dim,
                 fake_activation,
                 optimizer,
                 learning_rate,
                 adam_beta_1,
                 adam_beta_2,
                 batch_size,
                 epochs,
                 n_fid_samples,
                 tf_verbose):
        super(BaseGAN, self).__init__(tf_verbose)

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

        self._fid = _FrechetInceptionDistance(inception_input_shape=(96, 96, 3), batch_size=500)

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

    def _random_sampling_from_real_data(self, x, y=None, num_classes=None):
        idx = np.random.randint(0, x.shape[0], self.n_fid_samples)
        x = x[idx]

        if y is None:

            return x
        else:
            y = y[idx]
            selected_label = tf.convert_to_tensor(y, dtype=tf.dtypes.int32)
            selected_onehot = tf.keras.utils.to_categorical(selected_label, num_classes)

            return x, selected_onehot

    def _get_generated_image_for_fid(self, noise, onehot=None):
        generated_images = None
        if onehot is None:
            random_noise_set = tf.data.Dataset.from_tensor_slices(noise).batch(self.batch_size)
            for noise_batch in random_noise_set:
                generated_images_batch = self._gene(noise_batch, training=False).numpy()
                if generated_images is None:
                    generated_images = generated_images_batch
                else:
                    generated_images = np.vstack((generated_images, generated_images_batch))
        else:
            random_noise_set = tf.data.Dataset.from_tensor_slices((noise, onehot)).batch(self.batch_size)
            for noise_batch, onehot_batch in random_noise_set:
                generated_images_batch = self._gene([noise_batch, onehot_batch], training=False).numpy()
                if generated_images is None:
                    generated_images = generated_images_batch
                else:
                    generated_images = np.vstack((generated_images, generated_images_batch))

        return generated_images

    def _compute_frechet_inception_distance(self, fake_images):
        current_gene_weights = self._gene.get_weights()
        current_disc_weights = self._disc.get_weights()
        self._delete_session()
        fid = '%.2f' % self._fid.compute_fid(fake_images)
        self._create_session()
        self._gene = self._build_generator()
        self._gene.set_weights(current_gene_weights)
        self._disc = self._build_discriminator()
        self._disc.set_weights(current_disc_weights)

        return fid

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
