from transfer_gan.datasets import DatasetLoader
from transfer_gan.models.acgan import ACGANFashionMnist
from transfer_gan.utils.data_utils import get_data_information


if __name__ == '__main__':
    loader = DatasetLoader()
    (x_train, y_train), (x_test, y_test), class_names = loader.load_fashion_mnist()

    input_shape, num_classes = get_data_information(x_train, y_train)

    model = ACGANFashionMnist(
        input_shape=input_shape,
        num_classes=num_classes,
        noise_dim=100,
        fake_activation='tanh',
        batch_size=64,
        learning_rate=1e-4,
        adam_beta_1=0.9,
        epochs=15,
        n_fid_samples=0
    )
    model.fit(x_train, y_train, log_dir='log_acgan-fashion_mnist', log_period=1)
    model.predict(label=None, plot=True)
