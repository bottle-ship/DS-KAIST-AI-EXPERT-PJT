import os

import math
import numpy as np
import pandas as pd

import tensorflow as tf

from abc import abstractmethod
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras.utils import plot_model
from tqdm import trange

from ..utils.keras_utils import load_model_from_json, save_model_to_json
from ..utils.pickle_utils import load_from_pickle, save_to_pickle
from ..utils.visualization import show_generated_image

from ._base import BaseModel


class BaseACGAN(BaseModel):

    def __init__(self, input_shape,
                 num_classes,
                 noise_dim,
                 fake_activation='tanh',
                 batch_size=64,
                 learning_rate=1e-4,
                 beta_1=0.9,
                 epochs=15,
                 **kwargs):
        super(BaseACGAN, self).__init__(**kwargs)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        self.fake_activation = fake_activation
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.epochs = epochs

        self.input_channel_ = self.input_shape[-1]

        self._gene = self._build_generator()
        self._valid_generator_output_shape()
        self._disc = self._build_discriminator()

        self.generator_optimizer = self._set_optimizer()
        self.discriminator_optimizer = self._set_optimizer()

        self.history = list()

    def _initialize(self):
        self.input_channel_ = self.input_shape[-1]

        self.history.clear()

    def _valid_generator_output_shape(self):
        if not self.input_shape == self._gene.output_shape[1:]:
            raise ValueError(
                "Mismatch input shape(%s) and generator output shape(%s)" %
                (self.input_shape, self._gene.output_shape[1:])
            )

    @abstractmethod
    def _build_generator(self):
        raise NotImplementedError

    @abstractmethod
    def _build_discriminator(self):
        raise NotImplementedError

    def _set_optimizer(self):
        return tf.keras.optimizers.Adam(lr=self.learning_rate, beta_1=self.beta_1)

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

    def _apply_gradients(self, grad_discriminator, grad_generator):
        self.discriminator_optimizer.apply_gradients(zip(grad_discriminator, self._disc.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(grad_generator, self._gene.trainable_variables))

    def fit(self, x, y, n_gen_sample=25, log_dir=None, log_period=5):
        self._initialize()

        if log_dir is not None:
            if os.path.exists(log_dir):
                raise FileExistsError("'%s' is already exists." % log_dir)
            else:
                os.mkdir(log_dir)

        if self.fake_activation == 'sigmoid':
            scaled_x = x / 255.0
        elif self.fake_activation == 'tanh':
            scaled_x = x / 255.0
            scaled_x = (scaled_x * 2.) - 1.
        else:
            self.fake_activation = 'tanh'
            scaled_x = x / 255.0
            scaled_x = (scaled_x * 2.) - 1.

        ds_train = tf.data.Dataset.from_tensor_slices((scaled_x, y)).shuffle(scaled_x.shape[0]).batch(self.batch_size)

        random_vector = tf.random.normal([n_gen_sample, self.noise_dim])
        random_cls = tf.random.uniform([n_gen_sample, ], 0, self.num_classes, dtype=tf.dtypes.int32)
        test_cls_onehot = tf.keras.utils.to_categorical(random_cls, self.num_classes)

        for epoch in range(1, self.epochs + 1):
            epoch_loss_disc = list()
            epoch_loss_gene = list()

            tqdm_range = trange(math.ceil(x.shape[0] / self.batch_size))
            for (x_tr_batch, y_tr_batch), _ in zip(ds_train, tqdm_range):
                grad_disc, grad_gene, loss_disc, loss_gene = self._compute_gradients(
                    x_tr_batch, y_tr_batch
                )
                self._apply_gradients(grad_disc, grad_gene)

                epoch_loss_disc.append(loss_disc)
                epoch_loss_gene.append(loss_gene)

                tqdm_range.set_postfix_str(
                    "[Epoch] %05d [Loss Disc] %.3f [Loss Gene] %.3f" %
                    (epoch, np.array(epoch_loss_disc).mean(), np.array(epoch_loss_gene).mean())
                )
            tqdm_range.close()

            self.history.append([epoch, np.array(epoch_loss_disc).mean(), np.array(epoch_loss_gene).mean()])

            if log_dir is not None and epoch % log_period == 0:
                self.save_model(model_dir_name=os.path.join(log_dir, 'epoch_%05d' % epoch))

                gene_img = self._gene([random_vector, test_cls_onehot], training=False).numpy()
                if self.fake_activation == 'tanh':
                    gene_img = gene_img / 2 + 0.5

                show_generated_image(gene_img, filename=os.path.join(log_dir, 'epoch_%05d.png' % epoch))

    def predict(self, label=None, n_gen_sample=25, plot=False, filename=None):
        if label is not None and label >= self.num_classes:
            raise ValueError("The label must be < %d" % self.num_classes)

        random_vector = tf.random.normal([n_gen_sample, self.noise_dim])
        if label is None:
            random_cls = tf.random.uniform([n_gen_sample, ], 0, self.num_classes, dtype=tf.dtypes.int32)
        else:
            random_cls = tf.random.uniform([n_gen_sample, ], label, label + 1, dtype=tf.dtypes.int32)

        test_cls_onehot = tf.keras.utils.to_categorical(random_cls, self.num_classes)

        gene_img = self._gene([random_vector, test_cls_onehot], training=False).numpy()

        if self.fake_activation == 'tanh':
            gene_img = gene_img / 2 + 0.5

        if plot:
            show_generated_image(gene_img, filename=filename)

        return gene_img, random_cls

    def save_model(self, model_dir_name=None):
        if os.path.exists(model_dir_name):
            raise FileExistsError("'%s' is already exists." % model_dir_name)
        else:
            os.mkdir(model_dir_name)

        save_to_pickle(self.get_params(), os.path.join(model_dir_name, 'params.pkl'))

        save_model_to_json(self._gene, os.path.join(model_dir_name, 'generator_model.json'))
        self._gene.save_weights(os.path.join(model_dir_name, 'generator_weights.h5'))

        save_model_to_json(self._disc, os.path.join(model_dir_name, 'discriminator_model.json'))
        self._disc.save_weights(os.path.join(model_dir_name, 'discriminator_weights.h5'))

        pd.DataFrame(self.history, columns=['Epochs', 'Loss_Disc', 'Loss_Gene']).to_pickle(
            os.path.join(model_dir_name, 'history.pkl')
        )

    def load_model(self, model_dir_name=None):
        self.set_params(**load_from_pickle(os.path.join(model_dir_name, 'params.pkl')))

        self._initialize()

        self._gene = load_model_from_json(os.path.join(model_dir_name, 'generator_model.json'))
        self._gene.load_weights(os.path.join(model_dir_name, 'generator_weights.h5'))

        self._disc = load_model_from_json(os.path.join(model_dir_name, 'discriminator_model.json'))
        self._disc.load_weights(os.path.join(model_dir_name, 'discriminator_weights.h5'))

        self.history = pd.read_pickle(os.path.join(model_dir_name, 'history.pkl')).values.tolist()

    def show_generator_model(self, filename='generator.png'):
        self._gene.summary()
        plot_model(self._gene, filename)

    def show_discriminator_model(self, filename='discriminator.png'):
        self._disc.summary()
        plot_model(self._disc, filename)


class ACGANFashionMnist(BaseACGAN):

    def __init__(self, input_shape,
                 num_classes,
                 noise_dim,
                 fake_activation='tanh',
                 batch_size=64,
                 learning_rate=1e-4,
                 beta_1=0.9,
                 epochs=15,
                 **kwargs):
        super(ACGANFashionMnist, self).__init__(
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

        disc = layers.Dense(1, name='discriminator')(features)
        aux = layers.Dense(self.num_classes, name='auxiliary')(features)

        return tf.keras.Model(image, [disc, aux])


class ACGANCifar10(BaseACGAN):

    def __init__(self, input_shape,
                 num_classes,
                 noise_dim,
                 fake_activation='tanh',
                 batch_size=64,
                 learning_rate=1e-4,
                 beta_1=0.9,
                 epochs=15,
                 **kwargs):
        super(ACGANCifar10, self).__init__(
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

        disc = layers.Dense(1, name='discriminator')(features)
        aux = layers.Dense(self.num_classes, name='auxiliary')(features)

        return tf.keras.Model(image, [disc, aux])
