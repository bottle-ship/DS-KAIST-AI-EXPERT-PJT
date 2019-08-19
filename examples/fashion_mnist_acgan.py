from transfer_gan.datasets import DatasetLoader
from transfer_gan.datasets.fid_stats import get_fid_stats_path_fashion_mnist
from transfer_gan.models.acgan import ACGANFashionMnist
from transfer_gan.utils.data_utils import get_data_information


if __name__ == '__main__':
    loader = DatasetLoader()
    (x_train, y_train), (x_test, y_test), class_names = loader.load_fashion_mnist()

    input_shape, num_classes = get_data_information(x_train, y_train)

    model = ACGANFashionMnist(
        input_shape=input_shape,
        noise_dim=100,
        num_classes=num_classes,
        fake_activation='tanh',
        batch_size=64,
        optimizer='adam',
        learning_rate=1e-4,
        epochs=15,
        period_update_gene=1,
        n_fid_samples=5000,
        tf_verbose=False
    )
    model.set_fid_stats_path(get_fid_stats_path_fashion_mnist())
    model.fit(x_train, y_train, log_dir='log_fashion_mnist_acgan', log_period=1)
    model.predict(plot=True)
