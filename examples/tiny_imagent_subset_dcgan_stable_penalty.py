from transfer_gan.datasets import DatasetLoader
from transfer_gan.datasets.fid_stats import get_fid_stats_path_tiny_imagenet_subset
from transfer_gan.stable_models.dcgan_l1_penalty import DCGANTinyImagenetSubset
from transfer_gan.utils.data_utils import get_data_information


def main():
    loader = DatasetLoader()
    (x_train, y_train), (_, _), class_names = loader.load_tiny_imagenet_subset()

    input_shape, num_classes = get_data_information(x_train, y_train)

    model = DCGANTinyImagenetSubset(
        input_shape=input_shape,
        latent_dim=100,
        batch_size=100,
        fake_activation='tanh',
        learning_rate=0.0002,
        adam_beta_1=0.5,
        iterations=50000,
        fid_stats_path=get_fid_stats_path_tiny_imagenet_subset(),
        n_fid_samples=5000,
        tf_verbose=False
    )
    model.fit(x_train, log_dir='tiny_imagenet_subset_dcgan_stable_penalty', save_interval=200)


if __name__ == '__main__':
    main()
