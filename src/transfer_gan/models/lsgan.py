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


class BaseLSGAN(BaseGAN):

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
        super(BaseLSGAN, self).__init__(
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
            tf_verbose=tf_verbose
        )

    @abstractmethod
    def _build_generator(self):
        raise NotImplementedError

    @abstractmethod
    def _build_discriminator(self):
        raise NotImplementedError

    def _compute_loss(self, x):
        _mse = losses.MeanSquaredError()

        def _loss_generator(_fake_output):
            return _mse(tf.ones_like(_fake_output), _fake_output)

        def _loss_discriminator(_real_output, _fake_output):
            real_loss = _mse(tf.ones_like(_real_output), _real_output)
            fake_loss = _mse(tf.zeros_like(_fake_output), _fake_output)
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

    def _set_random_noise(self, n_image=None):
        if n_image is None:
            n_image = self.batch_size

        random_noise = tf.random.normal([n_image, self.noise_dim])

        return random_noise

    def fit(self, x, log_dir=None, log_period=5):
        self._initialize()

        if log_dir is not None:
            log_dir = make_directory(log_dir, time_suffix=True)

        scaled_x = self._scaling_image(x)

        if self.n_fid_samples > 0:
            fid_random_noise = tf.random.normal([self.n_fid_samples, self.noise_dim])
            selected_images = self._random_sampling_from_real_data(scaled_x)
            self._fid.compute_real_image_mean_and_cov(selected_images)
        else:
            fid_random_noise = None

        ds_train = tf.data.Dataset.from_tensor_slices(scaled_x).shuffle(scaled_x.shape[0]).batch(self.batch_size)

        random_noise = self._set_random_noise()

        for epoch in range(1, self.epochs + 1):
            epoch_loss_disc = list()
            epoch_loss_gene = list()

            fid = '-'
            iterations = int(np.ceil(x.shape[0] / self.batch_size))
            tqdm_range = trange(iterations)
            iter_cnt = 0
            for x_tr_batch, _ in zip(ds_train, tqdm_range):
                iter_cnt += 1
                grad_disc, grad_gene, loss_disc, loss_gene = self._compute_gradients(x_tr_batch)
                self._apply_gradients_discriminator(grad_disc)
                self._apply_gradients_generator(grad_gene)

                epoch_loss_disc.append(loss_disc)
                epoch_loss_gene.append(loss_gene)

                if iterations == iter_cnt and self.n_fid_samples > 0:
                    fid_generated_images = self._get_generated_image_for_fid(fid_random_noise)
                    fid = self._compute_frechet_inception_distance(fid_generated_images)

                tqdm_range.set_postfix_str(
                    "[Epoch] %05d [Loss Disc] %.3f [Loss Gene] %.3f [FID] %s" %
                    (epoch, np.array(epoch_loss_disc).mean(), np.array(epoch_loss_gene).mean(), fid)
                )
            tqdm_range.close()

            self.history.append([epoch, np.array(epoch_loss_disc).mean(), np.array(epoch_loss_gene).mean(), fid])

            if log_dir is not None and epoch % log_period == 0:
                self.save_model(model_dir_name=os.path.join(log_dir, 'epoch_%05d' % epoch))
                gene_img = self._gene(random_noise, training=False).numpy()
                gene_img = self._unscaling_image(gene_img)
                show_generated_image(gene_img, filename=os.path.join(log_dir, 'epoch_%05d_fid_%s.png' % (epoch, fid)))

    def predict(self, n_image=25, plot=False, filename=None):
        random_noise = self._set_random_noise(n_image=n_image)

        gene_img = self._gene(random_noise, training=False).numpy()
        gene_img = self._unscaling_image(gene_img)

        if plot:
            show_generated_image(gene_img, filename=filename)

        return gene_img


class LSGANMnist(BaseLSGAN):

    def __init__(self, input_shape,
                 noise_dim=100,
                 fake_activation='tanh',
                 optimizer='adam',
                 learning_rate=2e-4,
                 adam_beta_1=0.5,
                 adam_beta_2=0.999,
                 batch_size=32,
                 epochs=15,
                 n_fid_samples=5000,
                 tf_verbose=True,
                 **kwargs):
        super(LSGANMnist, self).__init__(
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
            tf_verbose=tf_verbose
        )
        self.kwargs = kwargs

    def _build_generator(self):
        inputs = layers.Input(shape=(self.noise_dim,))

        x = layers.Dense(256)(inputs)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization(momentum=0.8)(x)

        x = layers.Dense(512)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization(momentum=0.8)(x)

        x = layers.Dense(1024)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization(momentum=0.8)(x)

        x = layers.Dense(np.prod(self.input_shape))(x)

        fake = layers.Activation(self.fake_activation)(x)
        fake = layers.Reshape(self.input_shape)(fake)

        return tf.keras.Model(inputs, fake)

    def _build_discriminator(self):
        image = layers.Input(shape=self.input_shape)

        x = layers.Flatten()(image)

        x = layers.Dense(512)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

        x = layers.Dense(256)(x)
        features = layers.LeakyReLU(alpha=0.2)(x)

        validity = layers.Dense(1, name='discriminator')(features)

        return tf.keras.Model(image, validity)
