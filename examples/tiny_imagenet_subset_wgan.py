import tensorflow as tf

from tensorflow.python.keras import layers

from transfer_gan.datasets import DatasetLoader
from transfer_gan.models.wgan import BaseWGAN
from transfer_gan.utils.data_utils import get_data_information


class WGANTinyImagenetSubset(BaseWGAN):

    def __init__(self, input_shape,
                 noise_dim,
                 batch_size=64,
                 fake_activation='tanh',
                 optimizer='adam',
                 learning_rate=1e-4,
                 disc_clip_value=0.01,
                 epochs=15,
                 period_update_gene=3,
                 n_fid_samples=5000,
                 tf_verbose=False,
                 **kwargs):
        super(WGANTinyImagenetSubset, self).__init__(
            input_shape=input_shape,
            noise_dim=noise_dim,
            batch_size=batch_size,
            fake_activation=fake_activation,
            optimizer=optimizer,
            learning_rate=learning_rate,
            disc_clip_value=disc_clip_value,
            epochs=epochs,
            period_update_gene=period_update_gene,
            n_fid_samples=n_fid_samples,
            tf_verbose=tf_verbose,
            kwargs=kwargs
        )

    def _build_generator(self):
        inputs = layers.Input(shape=(self.noise_dim,))

        x = layers.Dense(8 * 8 * 256, use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Reshape((8, 8, 256))(x)

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

        x = layers.Conv2D(16, kernel_size=3, strides=2, padding='same',
                          kernel_initializer='glorot_normal', bias_initializer='Zeros')(image)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Conv2D(32, kernel_size=3, strides=1, padding='same',
                          kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same',
                          kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same',
                          kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Conv2D(256, kernel_size=3, strides=2, padding='same',
                          kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Conv2D(512, kernel_size=3, strides=1, padding='same',
                          kernel_initializer='glorot_normal', bias_initializer='Zeros')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.5)(x)

        features = layers.Flatten()(x)

        validity = layers.Dense(1, name='discriminator')(features)

        return tf.keras.Model(image, validity)


def train():
    loader = DatasetLoader()
    (x_train, y_train), (_, _), class_names = loader.load_tiny_imagenet_subset()

    input_shape, num_classes = get_data_information(x_train, y_train)

    model = WGANTinyImagenetSubset(
        input_shape=input_shape,
        noise_dim=110,
        optimizer='RMSProp',
        epochs=3000,
        period_update_gene=3
    )

    model.fit(x_train, log_dir='log_tiny_imagenet_subset_wgan', log_period=10)


if __name__ == '__main__':
    train()
