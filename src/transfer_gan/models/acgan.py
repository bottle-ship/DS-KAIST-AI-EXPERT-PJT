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


class BaseACGAN(BaseGAN):

    def __init__(self, input_shape,
                 num_classes,
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
        self.num_classes = num_classes

        super(BaseACGAN, self).__init__(
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

    def _compute_loss(self, x, y):
        _cross_entropy = losses.BinaryCrossentropy(from_logits=True)
        _cls_cross_entropy = losses.CategoricalCrossentropy(from_logits=True)

        def _loss_generator(_fake_output):
            return _cross_entropy(tf.ones_like(_fake_output), _fake_output)

        def _loss_discriminator(_real_output, _fake_output):
            real_loss = _cross_entropy(tf.ones_like(_real_output), _real_output)
            fake_loss = _cross_entropy(tf.zeros_like(_fake_output), _fake_output)
            total_loss = real_loss + fake_loss

            return total_loss

        def _loss_class(_y_onehot, _real_label, _y_onehot_fake, _fake_label):
            cls_loss_real = _cls_cross_entropy(_y_onehot, _real_label)
            cls_loss_fake = _cls_cross_entropy(_y_onehot_fake, _fake_label)
            total_cls_loss = cls_loss_real + cls_loss_fake

            return total_cls_loss

        y_onehot = tf.keras.utils.to_categorical(y, self.num_classes)
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        y_fake = tf.random.uniform([self.batch_size, ], 0, self.num_classes, dtype=tf.dtypes.int32)
        y_onehot_fake = tf.keras.utils.to_categorical(y_fake, self.num_classes)

        generated_images = self._gene([noise, y_onehot_fake], training=True)

        real_output, real_label = self._disc(x, training=True)
        fake_output, fake_label = self._disc(generated_images, training=True)

        loss_discriminator = _loss_discriminator(real_output, fake_output)
        loss_generator = _loss_generator(fake_output)
        loss_class = _loss_class(y_onehot, real_label, y_onehot_fake, fake_label)

        loss_discriminator = loss_discriminator + loss_class
        loss_generator = loss_generator + loss_class

        return loss_discriminator, loss_generator

    def _compute_gradients(self, x, y):
        with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
            loss_discriminator, loss_generator = self._compute_loss(x, y)

            grad_discriminator = discriminator_tape.gradient(loss_discriminator, self._disc.trainable_variables)
            grad_generator = generator_tape.gradient(loss_generator, self._gene.trainable_variables)

        return grad_discriminator, grad_generator, loss_discriminator, loss_generator

    def _apply_gradients_discriminator(self, grad_discriminator):
        self._disc_optimizer.apply_gradients(zip(grad_discriminator, self._disc.trainable_variables))

    def _apply_gradients_generator(self, grad_generator):
        self._gene_optimizer.apply_gradients(zip(grad_generator, self._gene.trainable_variables))

    def _set_random_noise_and_onehot(self, n_image=None, label=None):
        if n_image is None:
            n_image = self.batch_size

        if label is None:
            min_label = 0
            max_label = self.num_classes
        else:
            min_label = label
            max_label = label + 1

        random_noise = tf.random.normal([n_image, self.noise_dim])
        random_label = tf.random.uniform([n_image, ], min_label, max_label, dtype=tf.dtypes.int32)
        random_onehot = tf.keras.utils.to_categorical(random_label, self.num_classes)

        return random_noise, random_onehot, random_label

    def fit(self, x, y, log_dir=None, log_period=5):
        self._initialize()

        if log_dir is not None:
            log_dir = make_directory(log_dir, time_suffix=True)

        scaled_x = self._scaling_image(x)

        if self.n_fid_samples > 0:
            fid_random_noise = tf.random.normal([self.n_fid_samples, self.noise_dim])
            selected_images, selected_onehot = self._random_sampling_from_real_data(scaled_x, y, self.num_classes)
            self._fid.compute_real_image_mean_and_cov(selected_images)
        else:
            fid_random_noise = None
            selected_onehot = None

        ds_train = tf.data.Dataset.from_tensor_slices((scaled_x, y)).shuffle(scaled_x.shape[0]).batch(self.batch_size)

        random_noise, random_onehot, _ = self._set_random_noise_and_onehot()

        for epoch in range(1, self.epochs + 1):
            epoch_loss_disc = list()
            epoch_loss_gene = list()

            fid = '-'
            iterations = int(np.ceil(x.shape[0] / self.batch_size))
            tqdm_range = trange(iterations)
            iter_cnt = 0
            for (x_tr_batch, y_tr_batch), _ in zip(ds_train, tqdm_range):
                iter_cnt += 1
                grad_disc, grad_gene, loss_disc, loss_gene = self._compute_gradients(
                    x_tr_batch, y_tr_batch
                )
                self._apply_gradients_discriminator(grad_disc)
                self._apply_gradients_generator(grad_gene)

                epoch_loss_disc.append(loss_disc)
                epoch_loss_gene.append(loss_gene)

                if iterations == iter_cnt and self.n_fid_samples > 0:
                    fid_generated_images = self._get_generated_image_for_fid(fid_random_noise, selected_onehot)
                    fid = self._compute_frechet_inception_distance(fid_generated_images)

                tqdm_range.set_postfix_str(
                    "[Epoch] %05d [Loss Disc] %.3f [Loss Gene] %.3f [FID] %s" %
                    (epoch, np.array(epoch_loss_disc).mean(), np.array(epoch_loss_gene).mean(), fid)
                )
            tqdm_range.close()

            self.history.append([epoch, np.array(epoch_loss_disc).mean(), np.array(epoch_loss_gene).mean(), fid])

            if log_dir is not None and epoch % log_period == 0:
                self.save_model(model_dir_name=os.path.join(log_dir, 'epoch_%05d' % epoch))
                gene_img = self._gene([random_noise, random_onehot], training=False).numpy()
                gene_img = self._unscaling_image(gene_img)
                show_generated_image(gene_img, filename=os.path.join(log_dir, 'epoch_%05d_fid_%s.png' % (epoch, fid)))

    def predict(self, label=None, n_image=25, plot=False, filename=None):
        if label is not None and label >= self.num_classes:
            raise ValueError("The label must be < %d" % self.num_classes)

        random_noise, random_onehot, random_label = self._set_random_noise_and_onehot(n_image=n_image, label=label)

        gene_img = self._gene([random_noise, random_onehot], training=False).numpy()
        gene_img = self._unscaling_image(gene_img)

        if plot:
            show_generated_image(gene_img, filename=filename)

        return gene_img, random_label.numpy()


class ACGANFashionMnist(BaseACGAN):

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
                 tf_verbose=True,
                 **kwargs):
        super(ACGANFashionMnist, self).__init__(
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
        self.kwargs = kwargs

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


class ACGANCifar10(BaseACGAN):

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
                 tf_verbose=True,
                 **kwargs):
        super(ACGANCifar10, self).__init__(
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
        self.kwargs = kwargs

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


class ACGANTinyImagenet(BaseACGAN):

    def __init__(self, input_shape,
                 num_classes,
                 noise_dim=110,
                 fake_activation='tanh',
                 optimizer='adam',
                 learning_rate=1e-4,
                 adam_beta_1=0.7,
                 adam_beta_2=0.999,
                 batch_size=64,
                 epochs=3000,
                 n_fid_samples=5000,
                 **kwargs):
        super(ACGANTinyImagenet, self).__init__(
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
            **kwargs
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
