from transfer_gan.datasets import DatasetLoader
from transfer_gan.datasets.fid_stats import get_fid_stats_path_tiny_imagenet_subset
from transfer_gan.stable_models.wgan_gp import WGANGPTinyImagenetSubset
from transfer_gan.utils.data_utils import get_data_information


def main():
    loader = DatasetLoader()
    (x_train, y_train), (_, _), class_names = loader.load_tiny_imagenet_subset()

    input_shape, num_classes = get_data_information(x_train, y_train)

    model = WGANGPTinyImagenetSubset(
        input_shape=input_shape,
        latent_dim=100,
        batch_size=128,
        fake_activation='tanh',
        learning_rate=0.0002,
        adam_beta_1=0,
        adam_beta_2=0.9,
        gp_loss_weight=10,
        iterations=50000,
        n_critic=1,
        fid_stats_path=get_fid_stats_path_tiny_imagenet_subset(),
        n_fid_samples=5000,
        disc_model_path=None,
        disc_weights_path=None,
        gene_model_path=None,
        gene_weights_path=None,
        tf_verbose=False
    )
    model.fit(x_train, log_dir='tiny_imagenet_subset_wgangp_stable', save_interval=100)


if __name__ == '__main__':
    main()
