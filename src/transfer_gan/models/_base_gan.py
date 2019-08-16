import os

import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from abc import abstractmethod
from tensorflow.python.keras import losses
from tensorflow.python.keras.utils import plot_model
from tqdm import trange

from ..metrics.fid import create_realdata_stats, fid_with_realdata_stats
from ..utils.keras_utils import load_model_from_json, save_model_to_json
from ..utils.os_utils import make_directory
from ..utils.pickle_utils import load_from_pickle, save_to_pickle
from ..utils.visualization import show_generated_image

from ._base import BaseModel


class BaseGAN(BaseModel):

    def __init__(self, input_shape,
                 noise_dim,
                 num_classes,
                 batch_size,
                 fake_activation,
                 optimizer,
                 learning_rate,
                 disc_clip_value,
                 epochs,
                 period_update_gene,
                 n_fid_samples,
                 tf_verbose,
                 **kwargs):
        super(BaseGAN, self).__init__(tf_verbose)

        self.input_shape = input_shape
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.fake_activation = fake_activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.disc_clip_value = disc_clip_value
        self.epochs = epochs
        self.period_update_gene = period_update_gene
        self.n_fid_samples = n_fid_samples
        self.kwargs = kwargs

        self.input_channel_ = self.input_shape[-1]

        self._gene = self._build_generator()
        self._validate_generator_output_shape()
        self._disc = self._build_discriminator()

        self._gene_optimizer = self._set_optimizer()
        self._disc_optimizer = self._set_optimizer()

        self._fid_stats_path = './fid_stats' + datetime.datetime.now().strftime("_%Y%m%d-%H%M%S") + '.npz'

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

    def _validate_generator_output_shape(self):
        if not self.input_shape == self._gene.output_shape[1:]:
            raise ValueError(
                "Mismatch input shape(%s) and generator output shape(%s)" %
                (self.input_shape, self._gene.output_shape[1:])
            )

    @abstractmethod
    def _build_generator(self):
        raise NotImplementedError

    @abstractmethod
    def _build_discriminator(self,):
        raise NotImplementedError

    @abstractmethod
    def _compute_loss_generator(self, fake_output):
        raise NotImplementedError

    @abstractmethod
    def _compute_loss_discriminator(self, real_output, fake_output):
        raise NotImplementedError

    @staticmethod
    def _compute_loss_class(y_onehot, y_label, y_fake_onehot, y_fake_label):
        loss_class_real = losses.CategoricalCrossentropy(from_logits=True)(y_onehot, y_label)
        loss_class_fake = losses.CategoricalCrossentropy(from_logits=True)(y_fake_onehot, y_fake_label)
        loss_class = loss_class_real + loss_class_fake

        return loss_class

    def _compute_gradients(self, x, y=None):
        with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
            noise = self._get_random_noise(self.batch_size)

            if y is not None:
                y_onehot = tf.keras.utils.to_categorical(y, self.num_classes)
                y_fake = tf.random.uniform([self.batch_size, ], 0, self.num_classes, dtype=tf.dtypes.int32)
                y_fake_onehot = tf.keras.utils.to_categorical(y_fake, self.num_classes)

                generated_images = self._gene([noise, y_fake_onehot], training=True)

                real_output, y_label = self._disc(x, training=True)
                fake_output, y_fake_label = self._disc(generated_images, training=True)

            else:
                generated_images = self._gene(noise, training=True)

                real_output = self._disc(x, training=True)
                fake_output = self._disc(generated_images, training=True)

            loss_discriminator = self._compute_loss_discriminator(real_output, fake_output)
            loss_generator = self._compute_loss_generator(fake_output)

            if y is not None:
                loss_class = self._compute_loss_class(y_onehot, y_label, y_fake_onehot, y_fake_label)
                loss_discriminator = loss_discriminator + loss_class
                loss_generator = loss_generator + loss_class

            grad_discriminator = discriminator_tape.gradient(loss_discriminator, self._disc.trainable_variables)
            grad_generator = generator_tape.gradient(loss_generator, self._gene.trainable_variables)

        return grad_discriminator, grad_generator, loss_discriminator, loss_generator

    def _apply_gradients_discriminator(self, grad_discriminator):
        self._disc_optimizer.apply_gradients(zip(grad_discriminator, self._disc.trainable_variables))

    def _apply_gradients_generator(self, grad_generator):
        self._gene_optimizer.apply_gradients(zip(grad_generator, self._gene.trainable_variables))

    def _set_optimizer(self):
        if self.optimizer.lower() == 'adam':
            beta_1 = 0.9
            if 'adam_beta_1' in self.kwargs.keys():
                beta_1 = self.kwargs['adam_beta_1']

            beta_2 = 0.999
            if 'adam_beta_2' in self.kwargs.keys():
                beta_1 = self.kwargs['adam_beta_2']

            optimizer = tf.keras.optimizers.Adam(
                lr=self.learning_rate, beta_1=beta_1, beta_2=beta_2
            )
        elif self.optimizer.lower() == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=self.learning_rate
            )
        else:
            raise ValueError("The optimizer '%s' is not supported." % self.optimizer)

        return optimizer

    def _get_random_noise(self, n_images=None):
        if n_images is None:
            n_images = self.batch_size

        random_noise = tf.random.normal([n_images, self.noise_dim])

        return random_noise

    def _get_random_label_and_onehot(self, n_images=None, label=None):
        if self.num_classes is not None:
            if n_images is None:
                n_images = self.batch_size

            if label is None:
                min_label = 0
                max_label = self.num_classes
            else:
                min_label = label
                max_label = label + 1

            random_label = tf.random.uniform([n_images, ], min_label, max_label, dtype=tf.dtypes.int32)
            random_onehot = tf.keras.utils.to_categorical(random_label, self.num_classes)
        else:
            random_label = None
            random_onehot = None

        return random_label, random_onehot

    def _get_random_sample_from_real_data(self, x, y=None, num_classes=None):
        idx = np.random.randint(0, x.shape[0], self.n_fid_samples)
        x = x[idx]

        if y is None:

            return x, None
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
                generated_images_batch = self._unscaling_image(generated_images_batch)
                if generated_images is None:
                    generated_images = generated_images_batch
                else:
                    generated_images = np.vstack((generated_images, generated_images_batch))
        else:
            random_noise_set = tf.data.Dataset.from_tensor_slices((noise, onehot)).batch(self.batch_size)
            for noise_batch, onehot_batch in random_noise_set:
                generated_images_batch = self._gene([noise_batch, onehot_batch], training=False).numpy()
                generated_images_batch = self._unscaling_image(generated_images_batch)
                if generated_images is None:
                    generated_images = generated_images_batch
                else:
                    generated_images = np.vstack((generated_images, generated_images_batch))

        return generated_images

    def _fit(self, x, y=None, log_dir=None, log_period=5):
        if y is None:
            flag_conditional = False
        else:
            flag_conditional = True

        self._initialize()

        if log_dir is not None:
            log_dir = make_directory(log_dir, time_suffix=True)

        if self.n_fid_samples > 0:
            if self.input_channel_ == 1:
                create_realdata_stats(np.repeat(x, 3, axis=3), self._fid_stats_path)
            else:
                create_realdata_stats(x, self._fid_stats_path)

        fid_random_noise = self._get_random_noise(n_images=self.n_fid_samples)
        _, fid_random_onehot = self._get_random_label_and_onehot(n_images=self.n_fid_samples, label=None)

        scaled_x = self._scaling_image(x)

        if not flag_conditional:
            y = np.empty((x.shape[0], ))
        ds_train = tf.data.Dataset.from_tensor_slices((scaled_x, y)).shuffle(scaled_x.shape[0]).batch(self.batch_size)

        ref_random_noise = self._get_random_noise()
        _, ref_random_onehot = self._get_random_label_and_onehot()

        for epoch in range(1, self.epochs + 1):
            epoch_loss_disc = list()
            epoch_loss_gene = list()

            fid = '-'
            iterations = int(np.ceil(x.shape[0] / self.batch_size))
            tqdm_range = trange(iterations)
            iter_cnt = 0

            for (x_tr_batch, y_tr_batch), _ in zip(ds_train, tqdm_range):
                iter_cnt += 1

                if not flag_conditional:
                    grad_disc, grad_gene, loss_disc, loss_gene = self._compute_gradients(x=x_tr_batch, y=None)
                else:
                    grad_disc, grad_gene, loss_disc, loss_gene = self._compute_gradients(x=x_tr_batch, y=y_tr_batch)

                self._apply_gradients_discriminator(grad_disc)
                epoch_loss_disc.append(loss_disc)

                if self.disc_clip_value is not None:
                    for layer in self._disc.layers:
                        disc_weights = layer.get_weights()
                        disc_weights = [
                            np.clip(w, -1 * self.disc_clip_value, self.disc_clip_value) for w in disc_weights
                        ]
                        layer.set_weights(disc_weights)

                if iter_cnt % self.period_update_gene == 0:
                    self._apply_gradients_generator(grad_gene)
                    epoch_loss_gene.append(loss_gene)

                if iterations == iter_cnt and self.n_fid_samples > 0:
                    fid_generated_images = self._get_generated_image_for_fid(fid_random_noise, fid_random_onehot)
                    if self.input_channel_ == 1:
                        fid_generated_images = np.repeat(fid_generated_images, 3, axis=3)
                    fid = fid_with_realdata_stats(fid_generated_images, self._fid_stats_path)

                if iter_cnt < self.period_update_gene:
                    tqdm_range.set_postfix_str(
                        "[Epoch] %05d [Loss Disc] %.3f [Loss Gene] - [FID] %s" %
                        (epoch, np.array(epoch_loss_disc).mean(), fid)
                    )
                else:
                    tqdm_range.set_postfix_str(
                        "[Epoch] %05d [Loss Disc] %.3f [Loss Gene] %.3f [FID] %s" %
                        (epoch, np.array(epoch_loss_disc).mean(), np.array(epoch_loss_gene).mean(), fid)
                    )
            tqdm_range.close()

            self.history.append([epoch, np.array(epoch_loss_disc).mean(), np.array(epoch_loss_gene).mean(), fid])

            if log_dir is not None and epoch % log_period == 0:
                self.save_model(model_dir_name=os.path.join(log_dir, 'epoch_%05d' % epoch))
                if not flag_conditional:
                    gene_img = self._gene(ref_random_noise, training=False).numpy()
                else:
                    gene_img = self._gene([ref_random_noise, ref_random_onehot], training=False).numpy()
                gene_img = self._unscaling_image(gene_img)
                show_generated_image(gene_img, filename=os.path.join(log_dir, 'epoch_%05d_fid_%s.png' % (epoch, fid)))

    def _predict(self, n_images=25, plot=False, filename=None):
        random_noise = self._get_random_noise(n_images=n_images)

        gene_img = self._gene(random_noise, training=False).numpy()
        gene_img = self._unscaling_image(gene_img)

        if plot:
            show_generated_image(gene_img, filename=filename)

        return gene_img

    def _predict_with_label(self, n_images=25, label=None, plot=False, filename=None):
        if label is not None and label >= self.num_classes:
            raise ValueError("The label must be < %d" % self.num_classes)

        random_noise = self._get_random_noise(n_images=n_images)
        random_label, random_onehot = self._get_random_label_and_onehot(n_images=n_images, label=label)

        gene_img = self._gene([random_noise, random_onehot], training=False).numpy()
        gene_img = self._unscaling_image(gene_img)

        if plot:
            show_generated_image(gene_img, filename=filename)

        return gene_img, random_label.numpy()

    @abstractmethod
    def fit(self, x, **kwargs):
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
