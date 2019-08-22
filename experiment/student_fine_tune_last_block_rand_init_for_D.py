import os

from transfer_gan.datasets import DatasetLoader
from transfer_gan.datasets.fid_stats import get_fid_stats_path_cifar10
from transfer_gan.stable_models.dcgan import DCGANTinyImagenetSubset
from transfer_gan.utils.data_utils import get_data_information


WEIGHT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'source_model', 'teacher_dcgan_tiny_imagenet_subset'
)


def main():
    loader = DatasetLoader()
    (x_train, y_train), (_, _), class_names = loader.load_cifar10_subset()

    input_shape, num_classes = get_data_information(x_train, y_train)

    model = DCGANTinyImagenetSubset(
        input_shape=input_shape,
        latent_dim=100,
        batch_size=128,
        fake_activation='tanh',
        learning_rate=0.0002,
        adam_beta_1=0.5,
        iterations=50000,
        fid_stats_path=get_fid_stats_path_cifar10(),
        n_fid_samples=5000,
        tf_verbose=False
    )
    initial_disc_weights = model.discriminator.get_weights()

    model.discriminator.load_weights(os.path.join(WEIGHT_PATH, 'discriminator_weights.h5'))
    model.generator.load_weights(os.path.join(WEIGHT_PATH, 'generator_weights.h5'))

    new_disc_weights = model.discriminator.get_weights()
    new_gene_weights = model.generator.get_weights()

    print("Discriminator's layer - %d" % len( model.discriminator.layers))
    for layer in model.discriminator.layers:
        print(layer)
    print()

    print("Generator's layer - %d" % len(model.generator.layers))
    for layer in model.generator.layers:
        print(layer)
    print()

    """
    Discriminator summary
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 32, 32, 3)         0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 16, 16, 128)       9728      0, 1
    _________________________________________________________________
    leaky_re_lu_1 (LeakyReLU)    (None, 16, 16, 128)       0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 16, 16, 128)       0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 16, 16, 256)       295168    2, 3
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 16, 16, 256)       1024      4, 5, 6, 7
    _________________________________________________________________
    leaky_re_lu_2 (LeakyReLU)    (None, 16, 16, 256)       0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 16, 16, 256)       0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 8, 8, 512)         1180160   8, 9
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 8, 8, 512)         2048      10, 11, 12, 13
    _________________________________________________________________
    leaky_re_lu_3 (LeakyReLU)    (None, 8, 8, 512)         0         
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 8, 8, 512)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 32768)             0         
    _________________________________________________________________
    discriminator (Dense)        (None, 1)                 32769     14, 15
    =================================================================
    Total params: 3,040,258
    Trainable params: 1,519,361
    Non-trainable params: 1,520,897
    """
    """
    0: (5, 5, 3, 128)
    1: (128,)
    2: (3, 3, 128, 256)
    3: (256,)
    4: (256,)
    5: (256,)
    6: (256,)
    7: (256,)
    8: (3, 3, 256, 512)
    9: (512,)
    10: (512,)
    11: (512,)
    12: (512,)
    13: (512,)
    14: (32768, 1)
    15: (1,)
    """

    for i in range(8, len(model.discriminator.layers)):
        new_disc_weights[i] = initial_disc_weights[i]

    model.discriminator.set_weights(new_disc_weights)
    model.generator.set_weights(new_gene_weights)

    model.compile_discriminator()
    model.compile_combine_model()

    model.fit(x_train, log_dir=__file__.split(os.sep)[-1].split('.')[0], save_interval=100)


if __name__ == '__main__':
    main()
