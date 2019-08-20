from transfer_gan.datasets import DatasetLoader
from transfer_gan.datasets.fid_stats import get_fid_stats_path_tiny_imagenet_subset
from transfer_gan.stable_models.acgan import ACGANTinyImagenetSubset
from transfer_gan.utils.data_utils import get_data_information


def main():
    loader = DatasetLoader()
    (x_train, y_train), (_, _), class_names = loader.load_tiny_imagenet_subset()

    input_shape, num_classes = get_data_information(x_train, y_train)

    model = ACGANTinyImagenetSubset(
        input_shape=input_shape,
        num_classes=num_classes,
        latent_dim=100,
        batch_size=128,
        fake_activation='tanh',
        learning_rate=0.0002,
        adam_beta_1=0.5,
        iterations=50000,
        fid_stats_path=get_fid_stats_path_tiny_imagenet_subset(),
        n_fid_samples=5000,
        tf_verbose=False
    )
    model.fit(x_train, y_train, log_dir='tiny_imagenet_subset_acgan_stable', save_interval=100)


if __name__ == '__main__':
    main()
