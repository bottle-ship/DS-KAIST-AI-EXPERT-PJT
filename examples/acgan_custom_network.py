import tensorflow as tf

from tensorflow.python.keras import layers

from transfer_gan.datasets import DatasetLoader
from transfer_gan.models.acgan import BaseACGAN
from transfer_gan.utils.data_utils import get_data_information


class ACGANCustom(BaseACGAN):

    def __init__(self, input_shape,
                 num_classes,
                 noise_dim,
                 fake_activation='tanh',
                 batch_size=64,
                 learning_rate=1e-4,
                 beta_1=0.9,
                 epochs=15,
                 **kwargs):
        super(ACGANCustom, self).__init__(
            input_shape=input_shape,
            num_classes=num_classes,
            noise_dim=noise_dim,
            fake_activation=fake_activation,
            batch_size=batch_size,
            learning_rate=learning_rate,
            beta_1=beta_1,
            epochs=epochs,
            **kwargs
        )

    def _build_generator(self):
        ##########################
        # Implementation code Here
        ##########################

        z = layers.Input(shape=(self.noise_dim,))
        y = layers.Input(shape=(self.num_classes,))

        inputs = layers.concatenate([z, y])

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

        return tf.keras.Model([z, y], fake)

    def _build_discriminator(self):
        ##########################
        # Implementation code Here
        ##########################

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

        disc = layers.Dense(1, name='discriminator')(features)
        aux = layers.Dense(self.num_classes, name='auxiliary')(features)

        return tf.keras.Model(image, [disc, aux])


if __name__ == '__main__':
    loader = DatasetLoader()
    (x_train, y_train), (x_test, y_test), class_names = loader.load_fashion_mnist()

    input_shape, num_classes = get_data_information(x_train, y_train)

    model = ACGANCustom(
        input_shape=input_shape,
        num_classes=num_classes,
        noise_dim=100,
        fake_activation='tanh',
        batch_size=64,
        learning_rate=1e-4,
        beta_1=0.9,
        epochs=15
    )
    model.fit(x_train, y_train, log_dir='log_fashion_mnist', log_period=1)
    model.predict(label=None, plot=True)
