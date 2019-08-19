import os

import numpy as np
import tensorflow as tf

from abc import abstractmethod
from keras import backend as K
from keras.layers import Input
from keras.optimizers import Adam
from keras import layers
from keras import models

from ..metrics.fid import fid_with_realdata_stats
from ..utils.keras_utils import load_model_from_json, save_model_to_json
from ..utils.os_utils import make_directory
from ..utils.visualization import show_generated_image


class BaseDCGAN(object):

    def __init__(self, input_shape,
                 latent_dim,
                 batch_size,
                 fake_activation,
                 learning_rate,
                 adam_beta_1,
                 iterations,
                 fid_stats_path,
                 n_fid_samples,
                 tf_verbose):
        if not tf_verbose:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.config = tf.compat.v1.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.333
        K.set_session(tf.compat.v1.Session(config=self.config))

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.fake_activation = fake_activation
        self.learning_rate = learning_rate
        self.adam_beta_1 = adam_beta_1
        self.iterations = iterations
        self.fid_stats_path = fid_stats_path
        self.n_fid_samples = n_fid_samples

        self.input_channel_ = self.input_shape[-1]

        self.optimizer = Adam(self.learning_rate, self.adam_beta_1)

        # Build and compile the discriminator
        self.discriminator = self._build_discriminator()
        # self.discriminator.compile(loss='binary_crossentropy',
        #                            optimizer=self.optimizer,
        #                            metrics=['accuracy'])
        self._compile_discriminator()

        # Build the generator
        self.generator = self._build_generator()
        self._validate_generator_output_shape()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = models.Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def _compile_discriminator(self):
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])

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

    def _validate_generator_output_shape(self):
        if not self.input_shape == self.generator.output_shape[1:]:
            raise ValueError(
                "Mismatch input shape(%s) and generator output shape(%s)" %
                (self.input_shape, self.generator.output_shape[1:])
            )

    @abstractmethod
    def _build_generator(self):
        raise NotImplementedError

    @abstractmethod
    def _build_discriminator(self, ):
        raise NotImplementedError

    def fit(self, x, log_dir=None, save_interval=50):
        if log_dir is not None:
            log_dir = make_directory(log_dir, time_suffix=True)

        scaled_x = self._scaling_image(x)

        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        ref_noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
        ref_fid_noise = np.random.normal(0, 1, (self.n_fid_samples, self.latent_dim))

        for iteration in range(self.iterations):
            idx = np.random.randint(0, scaled_x.shape[0], self.batch_size)
            imgs = scaled_x[idx]

            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = self.combined.train_on_batch(noise, valid)

            if iteration % save_interval == 0:
                gen_imgs = self.generator.predict(ref_fid_noise)
                gen_imgs = self._unscaling_image(gen_imgs)

                fid_score = fid_with_realdata_stats(gen_imgs, self.fid_stats_path)
                print("%d [D loss: %.3f, acc.: %.2f%%] [G loss: %.3f] [FID: %.2f]" %
                      (iteration, d_loss[0], 100 * d_loss[1], g_loss, fid_score))

                if log_dir is not None:
                    self.save_model(model_dir_name=os.path.join(log_dir, 'iteration_%05d' % iteration))
                    ref_gen_imgs = self.generator.predict(ref_noise)
                    ref_gen_imgs = self._unscaling_image(ref_gen_imgs)
                    show_generated_image(
                        ref_gen_imgs,
                        filename=os.path.join(log_dir, 'iteration_%05d_fid_%.2f.png' % (iteration, fid_score))
                    )

    def predict(self, n_images=25, plot=False, filename=None):
        noise = np.random.normal(0, 1, (n_images, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = self._unscaling_image(gen_imgs)

        if plot:
            show_generated_image(gen_imgs, filename=filename)

        return gen_imgs

    def save_model(self, model_dir_name):
        make_directory(model_dir_name)

        save_model_to_json(self.generator, os.path.join(model_dir_name, 'generator_model.json'))
        self.generator.save_weights(os.path.join(model_dir_name, 'generator_weights.h5'))

        save_model_to_json(self.discriminator, os.path.join(model_dir_name, 'discriminator_model.json'))
        self.discriminator.save_weights(os.path.join(model_dir_name, 'discriminator_weights.h5'))

    def load_model(self, model_dir_name):
        self.discriminator = load_model_from_json(os.path.join(model_dir_name, 'discriminator_model.json'))
        self.discriminator.load_weights(os.path.join(model_dir_name, 'discriminator_weights.h5'))

        self._compile_discriminator()

        self.generator = load_model_from_json(os.path.join(model_dir_name, 'generator_model.json'))
        self.generator.load_weights(os.path.join(model_dir_name, 'generator_weights.h5'))

        self._validate_generator_output_shape()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = models.Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)


class DCGANTinyImagenetSubset(BaseDCGAN):

    def __init__(self, input_shape,
                 latent_dim,
                 batch_size=128,
                 fake_activation='tanh',
                 learning_rate=0.0002,
                 adam_beta_1=0.5,
                 iterations=50000,
                 fid_stats_path=None,
                 n_fid_samples=5000,
                 tf_verbose=False):
        super(DCGANTinyImagenetSubset, self).__init__(
            input_shape=input_shape,
            latent_dim=latent_dim,
            batch_size=batch_size,
            fake_activation=fake_activation,
            learning_rate=learning_rate,
            adam_beta_1=adam_beta_1,
            iterations=iterations,
            fid_stats_path=fid_stats_path,
            n_fid_samples=n_fid_samples,
            tf_verbose=tf_verbose
        )

    def _build_generator(self):
        inputs = layers.Input(shape=(self.latent_dim,))

        x = layers.Dense(512 * 4 * 4)(inputs)
        x = layers.ReLU()(x)
        x = layers.Reshape((4, 4, 512))(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(256, kernel_size=3, padding="same")(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization(momentum=0.8)(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(128, kernel_size=3, padding="same")(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization(momentum=0.8)(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(self.input_channel_, kernel_size=3, padding="same")(x)

        fake = layers.Activation(self.fake_activation)(x)

        return models.Model(inputs, fake)

    def _build_discriminator(self):
        image = layers.Input(shape=self.input_shape)

        x = layers.Conv2D(128, kernel_size=5, strides=2, padding="same")(image)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(512, kernel_size=3, strides=2, padding="same")(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)

        features = layers.Flatten()(x)

        validity = layers.Dense(1, activation='sigmoid', name='discriminator')(features)

        return models.Model(image, validity)