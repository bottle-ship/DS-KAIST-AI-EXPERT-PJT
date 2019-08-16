import numpy as np
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
                 optimizer='adam',
                 learning_rate=1e-4,
                 adam_beta_1=0.9,
                 adam_beta_2=0.999,
                 batch_size=64,
                 epochs=15,
                 n_fid_samples=5000,
                 tf_verbose=False):
        super(ACGANCustom, self).__init__(
            input_shape=input_shape,
            num_classes=num_classes,
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

    def _build_generator(self):

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

        disc = layers.Dense(1, name='discriminator')(features)
        aux = layers.Dense(self.num_classes, name='auxiliary')(features)

        return tf.keras.Model(image, [disc, aux])


def train():
    loader = DatasetLoader()
    (x_train, y_train), (_, _), class_names = loader.load_tiny_imagenet_subset()

    # Data augmentation
    # Start
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0
    )
    datagen.fit(x_train)

    gen_cnt = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=128):
        x_train = np.vstack((x_train, x_batch))
        y_train = np.hstack((y_train, y_batch))

        gen_cnt += 1
        if gen_cnt > 100:
            break

    # End

    input_shape, num_classes = get_data_information(x_train, y_train)

    model = ACGANCustom(
        input_shape=input_shape,
        num_classes=num_classes,
        noise_dim=110,
        fake_activation='tanh',
        batch_size=64,
        learning_rate=1e-4,
        adam_beta_1=0.7,
        epochs=3000
    )

    model.fit(x_train, y_train, log_dir='log_tiny_imagenet_subset', log_period=20)
    model.predict(label=None, plot=True)


if __name__ == '__main__':
    train()
