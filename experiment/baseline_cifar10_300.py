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
    (x_train, y_train), (_, _), class_names = loader.load_cifar10_subset_300()

    input_shape, num_classes = get_data_information(x_train, y_train)

    model = DCGANTinyImagenetSubset(
        input_shape=input_shape,
        latent_dim=100,
        batch_size=128,
        fake_activation='tanh',
        learning_rate=0.00005,
        adam_beta_1=0.5,
        iterations=50000,
        fid_stats_path=get_fid_stats_path_cifar10(),
        n_fid_samples=5000,
        tf_verbose=False
    )

    model.fit(x_train, log_dir=__file__.split(os.sep)[-1].split('.')[0], save_interval=100)


if __name__ == '__main__':
    main()
