import os

import numpy as np
import pandas as pd
import tensorflow as tf

from abc import abstractmethod
from keras import losses
from keras import backend as K
from keras import layers
from keras import models
from keras.optimizers import Adam
from keras.utils import plot_model
import keras

from ..metrics.fid import fid_with_realdata_stats
from ..utils.keras_utils import load_model_from_json, save_model_to_json
from ..utils.os_utils import make_directory
from ..utils.visualization import show_generated_image
from keras.callbacks import LambdaCallback
from functools import partial
import os

def _get_currnet_file_path():
    return os.path.dirname(os.path.abspath(__file__))

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
                 disc_model_path,
                 disc_weights_path,
                 gene_model_path,
                 gene_weights_path,
                 tf_verbose):
        if not tf_verbose:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.regularierW = 0.001
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
        self.teacher_discriminator = self._build_teacher_discriminator()

        self.teacher_discriminator.load_weights('./discriminator_weights.h5')
        teacher_discriminator_weights_list = self.teacher_discriminator.get_weights()

        self.disc_dense_w = teacher_discriminator_weights_list[14]
        self.disc_conv1_w = teacher_discriminator_weights_list[0]
        self.disc_conv2_w = teacher_discriminator_weights_list[2]
        self.disc_conv3_w = teacher_discriminator_weights_list[8]
        '''
        flag = 0
        for teacher_weight_np in teacher_discriminator_weights_list:
            print(flag, ' : ', teacher_weight_np.shape)
            flag = flag + 1
        '''
        '''
        0  :  (5, 5, 3, 128)
        1  :  (128,)
        2  :  (3, 3, 128, 256)
        3  :  (256,)
        4  :  (256,)
        5  :  (256,)
        6  :  (256,)
        7  :  (256,)
        8  :  (3, 3, 256, 512)
        9  :  (512,)
        10  :  (512,)
        11  :  (512,)
        12  :  (512,)
        13  :  (512,)
        14  :  (32768, 1)
        15  :  (1,)
        '''


        if disc_model_path is None:
            self.discriminator = self._build_discriminator()
        else:
            self.discriminator = load_model_from_json(disc_model_path)
        self.discriminator.load_weights('./discriminator_weights.h5')




        #self.disc_conv1_w

        #init_weight_disc = self.discriminator.get_weights()
        #self.discriminator.load_weights('./discriminator_weights.h5')
        #self.teacher_weight_disc = self.discriminator.get_weights()
        #self.discriminator.set_weights(init_weight_disc)
        #self.teacher_weight_disc_np = np.array(self.teacher_weight_disc)

        #self.disc_weights_np = np.array(self.discriminator.get_weights())
        #self.diff_disc_weights = self.disc_weights_np - self.teacher_weight_disc_np
        #self.regularization_loss_disc = tf.add_n([tf.reduce_sum(tf.square(w)) for w in self.diff_disc_weights])

        self._compile_discriminator()

        if disc_weights_path is not None:
            self.discriminator.load_weights(disc_weights_path)

        # Build the generator
        self.teacher_generator = self._build_teacher_generator()
        self.teacher_generator.load_weights('./generator_weights.h5')
        teacher_generator_weights_list = self.teacher_generator.get_weights()

        self.gen_dense_w = teacher_generator_weights_list[0]
        self.gen_conv1_w = teacher_generator_weights_list[2]
        self.gen_conv2_w = teacher_generator_weights_list[8]
        self.gen_conv3_w = teacher_generator_weights_list[14]

        '''
        flag = 0
        for teacher_weight_np in teacher_generator_weights_list:
            print(flag, ' : ', teacher_weight_np.shape)
            flag = flag + 1
        '''
        '''
        0: (100, 8192)
        1: (8192,)
        2: (3, 3, 512, 256)
        3: (256,)
        4: (256,)
        5: (256,)
        6: (256,)
        7: (256,)
        8: (3, 3, 256, 128)
        9: (128,)
        10: (128,)
        11: (128,)
        12: (128,)
        13: (128,)
        14: (3, 3, 128, 3)
        15: (3,)
        '''

        if gene_model_path is None:
            self.generator = self._build_generator()
        else:
            self.generator = load_model_from_json(gene_model_path)
        self._validate_generator_output_shape()

        self.generator.load_weights('./generator_weights.h5')

        if gene_weights_path is not None:
            self.generator.load_weights(gene_weights_path)

        # The generator takes noise as input and generates imgs
        z = layers.Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = models.Model(z, valid)

        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        self.history = list()

    def _partial_binary_crossentropy(self, y_true, y_pred, _model):
        tmp = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        #_model.
        combined_weights_list = self.get_weights.weights

        loss = 0
        loss = tf.convert_to_tensor(loss, dtype=tf.float32)
        for combined_weights_np in w:
            # print(combined_weights_np.shape)
            combined_weights_tensor = tf.convert_to_tensor(combined_weights_np, dtype=tf.float32)
            regularization_loss_combined = tf.reduce_sum(tf.square(combined_weights_tensor))
            loss += regularization_loss_combined

        tmp = 0.0 * tmp + 1.0 * loss

        #tmp = tmp * 1.0 + tf.convert_to_tensor(w[1][1], dtype=tf.float32)

        return tmp

    def binary_crossentropy_gen(self, y_true, y_pred):
        tmp = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        tmp = tmp*0.0 + tf.convert_to_tensor(self.get_weights.weights[1][1], dtype=tf.float32)

        return tmp

    def _partial_regulization_gen(self, y_true, y_pred, w):
        loss = tf.convert_to_tensor(w[1][1], dtype=tf.float32)
        #loss = w[1][1]
        return loss

    def binary_crossentropy_disc(self, y_true, y_pred):
        tmp = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        #tmp = 0.0*tmp + 10.0*self.regularization_loss_disc
        tmp = tmp
        return tmp

    def _compile_discriminator(self):
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

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
    def _build_discriminator(self):
        raise NotImplementedError

    class weightHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.weights = []

        def on_train_end(self, batch, logs={}):
            #self.losses.append(logs.get('loss'))
            #self.weights.append(self.model.get_weights())
            self.weights = self.model.get_weights()

    def fit(self, x, log_dir=None, save_interval=50):
        if log_dir is not None:
            log_dir = make_directory(log_dir, time_suffix=True)
            self.show_discriminator_model(os.path.join(log_dir, 'discriminator.png'))
            self.show_generator_model(os.path.join(log_dir, 'generator.png'))

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

            #print("[Iter %05d] [D loss: %.3f, acc.: %.2f%%] [G loss: %.3f]" % (iteration, d_loss[0], 100 * d_loss[1], g_loss))

            #'''
            if iteration % save_interval == 0:
                gen_imgs = self.generator.predict(ref_fid_noise)
                gen_imgs = self._unscaling_image(gen_imgs)

                fid_score = fid_with_realdata_stats(gen_imgs, self.fid_stats_path)
                print("[Iter %05d] [D loss: %.3f, acc.: %.2f%%] [G loss: %.3f] [FID: %.2f]" %
                      (iteration, d_loss[0], 100 * d_loss[1], g_loss, fid_score))

                self.history.append([iteration, d_loss[0], g_loss, fid_score])

                if log_dir is not None:
                    self.save_model(model_dir_name=os.path.join(log_dir, 'iteration_%05d' % iteration))
                    pd.DataFrame(self.history, columns=['Epochs', 'Loss_Disc', 'Loss_Gene', 'FID']).to_csv(
                        os.path.join(log_dir, 'history.csv'), index=False
                    )
                    ref_gen_imgs = self.generator.predict(ref_noise)
                    ref_gen_imgs = self._unscaling_image(ref_gen_imgs)
                    show_generated_image(
                        ref_gen_imgs,
                        filename=os.path.join(log_dir, 'iteration_%05d_fid_%.2f.png' % (iteration, fid_score))
                    )
            #'''

    def predict(self, n_images=25, plot=False, filename=None):
        noise = np.random.normal(0, 1, (n_images, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = self._unscaling_image(gen_imgs)

        if plot:
            show_generated_image(gen_imgs, filename=filename)

        return gen_imgs

    def show_generator_model(self, filename='generator.png'):
        self.generator.summary()
        plot_model(self.generator, filename, show_shapes=True)

    def show_discriminator_model(self, filename='discriminator.png'):
        self.discriminator.summary()
        plot_model(self.discriminator, filename, show_shapes=True)

    def save_model(self, model_dir_name):
        make_directory(model_dir_name)

        save_model_to_json(self.generator, os.path.join(model_dir_name, 'generator_model.json'))
        self.generator.save_weights(os.path.join(model_dir_name, 'generator_weights.h5'))

        save_model_to_json(self.discriminator, os.path.join(model_dir_name, 'discriminator_model.json'))
        self.discriminator.save_weights(os.path.join(model_dir_name, 'discriminator_weights.h5'))


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
                 disc_model_path=None,
                 disc_weights_path=None,
                 gene_model_path=None,
                 gene_weights_path=None,
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
            disc_model_path=disc_model_path,
            disc_weights_path=disc_weights_path,
            gene_model_path=gene_model_path,
            gene_weights_path=gene_weights_path,
            tf_verbose=tf_verbose
        )

    from keras import backend as K

    def l1_reg(self, weight_matrix):
        return 0.01 * K.sum(K.abs(weight_matrix))

    def l2_reg(weight_matrix):
        return 0.01 * K.sum(K.sqrt(weight_matrix))


    def l1t_disc_dense(self, weight_matrix):
        #tmp = (weight_matrix - self.disc_dense_w)*(weight_matrix - self.disc_dense_w)
        #return self.regularierW * K.sum(K.sqrt(tmp))
        return self.regularierW * K.sum(K.abs(weight_matrix - self.disc_dense_w))

    def l1t_disc_conv1(self, weight_matrix):
        return self.regularierW * K.sum(K.abs(weight_matrix - self.disc_conv1_w))

    def l1t_disc_conv2(self, weight_matrix):
        return self.regularierW * K.sum(K.abs(weight_matrix - self.disc_conv2_w))

    def l1t_disc_conv3(self, weight_matrix):
        return self.regularierW * K.sum(K.abs(weight_matrix - self.disc_conv3_w))

    def l1t_gen_dense(self, weight_matrix):
        return self.regularierW * K.sum(K.abs(weight_matrix - self.gen_dense_w))

    def l1t_gen_conv1(self, weight_matrix):
        return self.regularierW * K.sum(K.abs(weight_matrix - self.gen_conv1_w))

    def l1t_gen_conv2(self, weight_matrix):
        return self.regularierW * K.sum(K.abs(weight_matrix - self.gen_conv2_w))

    def l1t_gen_conv3(self, weight_matrix):
        return self.regularierW * K.sum(K.abs(weight_matrix - self.gen_conv3_w))



    def _build_teacher_generator(self):
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

    def _build_teacher_discriminator(self):
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

    def _build_generator(self):
        inputs = layers.Input(shape=(self.latent_dim,))

        x = layers.Dense(512 * 4 * 4, kernel_regularizer=self.l1t_gen_dense)(inputs)
        x = layers.ReLU()(x)
        x = layers.Reshape((4, 4, 512))(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(256, kernel_size=3, padding="same", kernel_regularizer=self.l1t_gen_conv1)(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization(momentum=0.8)(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(128, kernel_size=3, padding="same", kernel_regularizer=self.l1t_gen_conv2)(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization(momentum=0.8)(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(self.input_channel_, kernel_size=3, padding="same", kernel_regularizer=self.l1t_gen_conv3)(x)

        fake = layers.Activation(self.fake_activation)(x)

        return models.Model(inputs, fake)

    def _build_discriminator(self):
        image = layers.Input(shape=self.input_shape)

        x = layers.Conv2D(128, kernel_size=5, strides=2, padding="same", kernel_regularizer=self.l1t_disc_conv1)(image)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(256, kernel_size=3, strides=1, padding="same", kernel_regularizer=self.l1t_disc_conv2)(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(512, kernel_size=3, strides=2, padding="same", kernel_regularizer=self.l1t_disc_conv3)(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)

        features = layers.Flatten()(x)

        validity = layers.Dense(1, activation='sigmoid', name='discriminator', kernel_regularizer=self.l1t_disc_dense)(features)

        return models.Model(image, validity)


#  https://github.com/khanrc/tf.gans-comparison/blob/master/models/wgan_gp.py

class DCGANTinyImagenetSubsetResidual(BaseDCGAN):

    def __init__(self, input_shape,
                 latent_dim,
                 batch_size=128,
                 fake_activation='tanh',
                 learning_rate=0.0002,
                 adam_beta_1=0.5,
                 iterations=50000,
                 fid_stats_path=None,
                 n_fid_samples=5000,
                 disc_model_path=None,
                 disc_weights_path=None,
                 gene_model_path=None,
                 gene_weights_path=None,
                 tf_verbose=False):
        super(DCGANTinyImagenetSubsetResidual, self).__init__(
            input_shape=input_shape,
            latent_dim=latent_dim,
            batch_size=batch_size,
            fake_activation=fake_activation,
            learning_rate=learning_rate,
            adam_beta_1=adam_beta_1,
            iterations=iterations,
            fid_stats_path=fid_stats_path,
            n_fid_samples=n_fid_samples,
            disc_model_path=disc_model_path,
            disc_weights_path=disc_weights_path,
            gene_model_path=gene_model_path,
            gene_weights_path=gene_weights_path,
            tf_verbose=tf_verbose
        )
        self.nf = self.input_shape[1]

    @staticmethod
    def _residual_block_down(inputs, outputs, kernel_size=(3, 3)):
        input_shape = inputs.shape
        print(inputs.shape)
        nf_input = input_shape[-1]

        shortcut = layers.AveragePooling2D(pool_size=(2, 2))(inputs)
        shortcut = layers.Conv2D(outputs, kernel_size=(1, 1))(shortcut)

        net = layers.ReLU()(inputs)
        # net = tf.keras.layers.BatchNormalization()(net)
        net = layers.Conv2D(nf_input, kernel_size=kernel_size)(net)
        net = layers.ReLU()(net)
        # net = tf.keras.layers.BatchNormalization()(net)
        net = layers.Conv2D(outputs, kernel_size=kernel_size)(net)
        net = layers.AveragePooling2D(pool_size=(2, 2))(net)

        return layers.Add()([shortcut, net])

    @staticmethod
    def _residual_block_up(inputs, outputs, kernel_size=(3, 3)):
        shortcut = layers.UpSampling2D()(inputs)
        shortcut = layers.Conv2D(outputs, kernel_size=(1, 1))(shortcut)

        net = layers.ReLU()(inputs)
        net = layers.BatchNormalization(momentum=0.8)(net)
        net = layers.UpSampling2D()(net)
        net = layers.Conv2D(outputs, kernel_size=kernel_size)(net)
        net = layers.ReLU()(net)
        net = layers.BatchNormalization(momentum=0.8)(net)
        net = layers.Conv2D(outputs, kernel_size=kernel_size)(net)

        return layers.Add()([shortcut, net])

    def _build_generator(self):
        nf = self.input_shape[1]

        inputs = layers.Input(shape=(self.latent_dim,))

        x = layers.Dense(8 * nf * 4 * 4)(inputs)
        x = layers.Reshape((4, 4, 8 * nf))(x)

        x = self._residual_block_up(x, 8 * nf)
        x = self._residual_block_up(x, 4 * nf)
        x = self._residual_block_up(x, 2 * nf)
        x = self._residual_block_up(x, 1 * nf)

        x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.Conv2D(self.input_channel_, kernel_size=3, padding="same")(x)

        fake = layers.Activation(self.fake_activation)(x)

        return models.Model(inputs, fake)

    def _build_discriminator(self):
        nf = self.input_shape[1]

        image = layers.Input(shape=self.input_shape)

        x = layers.Conv2D(nf, kernel_size=3)(image)

        x = self._residual_block_down(x, 2 * nf)
        x = self._residual_block_down(x, 4 * nf)
        x = self._residual_block_down(x, 8 * nf)
        x = self._residual_block_down(x, 8 * nf)

        features = layers.Flatten()(x)

        validity = layers.Dense(1, name='discriminator')(features)

        return models.Model(image, validity)
