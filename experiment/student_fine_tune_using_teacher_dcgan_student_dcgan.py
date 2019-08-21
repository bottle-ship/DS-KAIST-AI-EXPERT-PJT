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

    model.discriminator.load_weights(os.path.join(WEIGHT_PATH, 'discriminator_weights.h5'))
    model.generator.load_weights(os.path.join(WEIGHT_PATH, 'generator_weights.h5'))

    new_disc_weights = model.discriminator.get_weights()
    new_gene_weights = model.generator.get_weights()

    print("Print discriminator's layer")
    for layer in model.discriminator.layers:
        print(layer)
    print()

    print("Print generator's layer")
    for layer in model.generator.layers:
        print(layer)
    print()

    model.discriminator.set_weights(new_disc_weights)
    model.generator.set_weights(new_gene_weights)

    model.compile_discriminator()
    model.compile_combine_model()

    model.fit(x_train, log_dir=__file__.split(os.sep)[-1].split('.')[0], save_interval=100)


if __name__ == '__main__':
    main()
