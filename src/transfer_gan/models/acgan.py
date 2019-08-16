import tensorflow as tf

from abc import abstractmethod
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses

from ._base_gan import BaseGAN


class BaseACGAN(BaseGAN):

    def __init__(self, input_shape,
                 noise_dim,
                 num_classes,
                 batch_size,
                 fake_activation,
                 optimizer,
                 learning_rate,
                 epochs,
                 period_update_gene,
                 n_fid_samples,
                 tf_verbose,
                 **kwargs):
        super(BaseACGAN, self).__init__(
            input_shape=input_shape,
            noise_dim=noise_dim,
            num_classes=num_classes,
            batch_size=batch_size,
            fake_activation=fake_activation,
            optimizer=optimizer,
            learning_rate=learning_rate,
            disc_clip_value=None,
            epochs=epochs,
            period_update_gene=period_update_gene,
            n_fid_samples=n_fid_samples,
            tf_verbose=tf_verbose,
            kwargs=kwargs
        )

    @abstractmethod
    def _build_generator(self):
        raise NotImplementedError

    @abstractmethod
    def _build_discriminator(self):
        raise NotImplementedError

    def _compute_loss_generator(self, fake_output):
        valid = tf.ones_like(fake_output)

        loss = losses.BinaryCrossentropy(from_logits=True)(valid, fake_output)

        return loss

    def _compute_loss_discriminator(self, real_output, fake_output):
        valid = tf.ones_like(real_output)
        fake = tf.zeros_like(fake_output)

        loss_real = losses.BinaryCrossentropy(from_logits=True)(valid, real_output)
        loss_fake = losses.BinaryCrossentropy(from_logits=True)(fake, fake_output)

        loss = loss_real + loss_fake

        return loss

    def fit(self, x, y=None, log_dir=None, log_period=5):
        self._fit(x=x, y=y, log_dir=log_dir, log_period=log_period)

        raise self

    def predict(self, n_images=25, plot=False, filename=None):
        raise self._predict(n_images=n_images, plot=plot, filename=filename)


class ACGANFashionMnist(BaseACGAN):

    def __init__(self, input_shape,
                 noise_dim,
                 num_classes,
                 batch_size=64,
                 fake_activation='tanh',
                 optimizer='adam',
                 learning_rate=1e-4,
                 epochs=15,
                 period_update_gene=3,
                 n_fid_samples=5000,
                 tf_verbose=False,
                 **kwargs):
        super(ACGANFashionMnist, self).__init__(
            input_shape=input_shape,
            noise_dim=noise_dim,
            num_classes=num_classes,
            batch_size=batch_size,
            fake_activation=fake_activation,
            optimizer=optimizer,
            learning_rate=learning_rate,
            epochs=epochs,
            period_update_gene=period_update_gene,
            n_fid_samples=n_fid_samples,
            tf_verbose=tf_verbose,
            kwargs=kwargs
        )

    def _build_generator(self):
        z = layers.Input(shape=(self.noise_dim,))
        y = layers.Input(shape=(self.num_classes,))

        inputs = layers.concatenate([z, y])

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

        return tf.keras.Model([z, y], fake)

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
        aux = layers.Dense(self.num_classes, name='auxiliary')(features)

        return tf.keras.Model(image, [validity, aux])
