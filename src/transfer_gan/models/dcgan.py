import os

import numpy as np
import tensorflow as tf

from abc import abstractmethod
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tqdm import trange

from ..utils.os_utils import make_directory
from ..utils.visualization import show_generated_image

from ._base_gan import BaseGAN
from ._base_gan import compute_fid


class BaseDCGAN(BaseGAN):

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
        super(BaseDCGAN, self).__init__(
            input_shape=input_shape,
            noise_dim=noise_dim,
            fake_activation=fake_activation,
            optimizer=optimizer,
            learning_rate=learning_rate,
            adam_beta_1=adam_beta_1,
            adam_beta_2=adam_beta_2,
            batch_size=batch_size,
            epochs=epochs,
            n_fid_samples=n_fid_samples,
            **kwargs
        )

    @abstractmethod
    def _build_generator(self):
        raise NotImplementedError

    @abstractmethod
    def _build_discriminator(self):
        raise NotImplementedError

    def _compute_loss(self, x):
        _cross_entropy = losses.BinaryCrossentropy(from_logits=True)

        def _loss_generator(_fake_output):
            return _cross_entropy(tf.ones_like(_fake_output), _fake_output)

        def _loss_discriminator(_real_output, _fake_output):
            real_loss = _cross_entropy(tf.ones_like(_real_output), _real_output)
            fake_loss = _cross_entropy(tf.zeros_like(_fake_output), _fake_output)
            total_loss = real_loss + fake_loss

            return total_loss

        noise = tf.random.normal([self.batch_size, self.noise_dim])

        generated_images = self._gene(noise, training=True)

        real_output = self._disc(x, training=True)
        fake_output = self._disc(generated_images, training=True)

        loss_discriminator = _loss_discriminator(real_output, fake_output)
        loss_generator = _loss_generator(fake_output)

        return loss_discriminator, loss_generator

    def _compute_gradients(self, x):
        with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
            loss_discriminator, loss_generator = self._compute_loss(x)

            grad_discriminator = discriminator_tape.gradient(loss_discriminator, self._disc.trainable_variables)
            grad_generator = generator_tape.gradient(loss_generator, self._gene.trainable_variables)

        return grad_discriminator, grad_generator, loss_discriminator, loss_generator

    def _apply_gradients_discriminator(self, grad_discriminator):
        self._disc_optimizer.apply_gradients(zip(grad_discriminator, self._disc.trainable_variables))

    def _apply_gradients_generator(self, grad_generator):
        self._gene_optimizer.apply_gradients(zip(grad_generator, self._gene.trainable_variables))

    def _random_sampling_from_real_data(self, x):
        idx = np.random.randint(0, x.shape[0], self.n_fid_samples)
        x = x[idx]

        return x

    def _set_random_noise(self, n_image=None):
        if n_image is None:
            n_image = self.batch_size

        random_noise = tf.random.normal([n_image, self.noise_dim])

        return random_noise

    def fit(self, x, log_dir=None, log_period=5):
        self._initialize()

        if log_dir is not None:
            make_directory(log_dir, time_suffix=True)

        scaled_x = self._scaling_image(x)

        if self.n_fid_samples > 0:
            selected_images = self._random_sampling_from_real_data(scaled_x)
            fid_random_noise = tf.random.normal([self.n_fid_samples, self.noise_dim])
            real_mean, real_cov = self._compute_image_mean_and_cov(selected_images)
        else:
            fid_random_noise = None
            real_mean = None
            real_cov = None

        ds_train = tf.data.Dataset.from_tensor_slices(scaled_x).shuffle(scaled_x.shape[0]).batch(self.batch_size)

        random_noise = self._set_random_noise()

        for epoch in range(1, self.epochs + 1):
            epoch_loss_disc = list()
            epoch_loss_gene = list()

            tqdm_range = trange(int(np.ceil(x.shape[0] / self.batch_size)))
            for x_tr_batch, _ in zip(ds_train, tqdm_range):
                grad_disc, grad_gene, loss_disc, loss_gene = self._compute_gradients(x_tr_batch)
                self._apply_gradients_discriminator(grad_disc)
                self._apply_gradients_generator(grad_gene)

                epoch_loss_disc.append(loss_disc)
                epoch_loss_gene.append(loss_gene)

                tqdm_range.set_postfix_str(
                    "[Epoch] %05d [Loss Disc] %.3f [Loss Gene] %.3f" %
                    (epoch, np.array(epoch_loss_disc).mean(), np.array(epoch_loss_gene).mean())
                )
            tqdm_range.close()

            if self.n_fid_samples > 0:
                fid_gene_images = self._gene(fid_random_noise, training=False).numpy()
                fid_fake_mean, fid_fake_cov = self._compute_image_mean_and_cov(fid_gene_images)
                fid = compute_fid(real_mean, real_cov, fid_fake_mean, fid_fake_cov)
                print("FID: %.2f" % fid)
            else:
                fid = '-'

            self.history.append([epoch, np.array(epoch_loss_disc).mean(), np.array(epoch_loss_gene).mean(), fid])

            if log_dir is not None and epoch % log_period == 0:
                self.save_model(model_dir_name=os.path.join(log_dir, 'epoch_%05d' % epoch))
                gene_img = self._gene(random_noise, training=False).numpy()
                gene_img = self._unscaling_image(gene_img)
                show_generated_image(gene_img, filename=os.path.join(log_dir, 'epoch_%05d.png' % epoch))

    def predict(self, n_image=25, plot=False, filename=None):
        random_noise = self._set_random_noise(n_image=n_image)

        gene_img = self._gene(random_noise, training=False).numpy()
        gene_img = self._unscaling_image(gene_img)

        if plot:
            show_generated_image(gene_img, filename=filename)

        return gene_img


class DCGANFashionMnist(BaseDCGAN):

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
        super(DCGANFashionMnist, self).__init__(
            input_shape=input_shape,
            noise_dim=noise_dim,
            fake_activation=fake_activation,
            optimizer=optimizer,
            learning_rate=learning_rate,
            adam_beta_1=adam_beta_1,
            adam_beta_2=adam_beta_2,
            batch_size=batch_size,
            epochs=epochs,
            n_fid_samples=n_fid_samples,
            **kwargs
        )

    def _build_generator(self):
        inputs = layers.Input(shape=(self.noise_dim,))

        x = layers.Dense(7 * 7 * 256, use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Reshape((7, 7, 256))(x)

        x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2DTranspose(self.input_channel_, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)

        fake = layers.Activation(self.fake_activation)(x)

        return tf.keras.Model(inputs, fake)

    def _build_discriminator(self):
        image = layers.Input(shape=self.input_shape)

        x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(image)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        features = layers.Flatten()(x)

        validity = layers.Dense(1, name='discriminator')(features)

        return tf.keras.Model(image, validity)
