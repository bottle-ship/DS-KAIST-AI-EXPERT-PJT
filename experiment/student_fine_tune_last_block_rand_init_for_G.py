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
    initial_gene_weights = model.generator.get_weights()

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

    model.generator.summary()
    """
    Generator summary
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)         (None, 100)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 8192)              827392    0, 1
    _________________________________________________________________
    re_lu_1 (ReLU)               (None, 8192)              0         
    _________________________________________________________________
    reshape_1 (Reshape)          (None, 4, 4, 512)         0         
    _________________________________________________________________
    up_sampling2d_1 (UpSampling2 (None, 8, 8, 512)         0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 8, 8, 256)         1179904   2, 3
    _________________________________________________________________
    re_lu_2 (ReLU)               (None, 8, 8, 256)         0         
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 8, 8, 256)         1024      4, 5, 6, 7
    _________________________________________________________________
    up_sampling2d_2 (UpSampling2 (None, 16, 16, 256)       0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 16, 16, 128)       295040    8, 9,
    _________________________________________________________________
    re_lu_3 (ReLU)               (None, 16, 16, 128)       0         
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 16, 16, 128)       512       10, 11, 12, 13
    _________________________________________________________________
    up_sampling2d_3 (UpSampling2 (None, 32, 32, 128)       0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 32, 32, 3)         3459      14, 15
    _________________________________________________________________
    activation_1 (Activation)    (None, 32, 32, 3)         0         
    =================================================================
    Total params: 2,307,331
    Trainable params: 2,306,563
    Non-trainable params: 768
    _________________________________________________________________

    """
    for w in new_gene_weights:
        print(w.shape)

    for i in [-2, -1]:
        new_gene_weights[i] = initial_gene_weights[i]

    model.discriminator.set_weights(new_disc_weights)
    model.generator.set_weights(new_gene_weights)

    model.compile_discriminator()
    model.compile_combine_model()

    model.fit(x_train, log_dir=__file__.split(os.sep)[-1].split('.')[0], save_interval=100)


if __name__ == '__main__':
    main()
