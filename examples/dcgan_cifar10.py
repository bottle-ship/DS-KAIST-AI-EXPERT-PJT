from transfer_gan.datasets import DatasetLoader
from transfer_gan.models.dcgan import DCGANCifar10
from transfer_gan.utils.data_utils import get_data_information


if __name__ == '__main__':
    loader = DatasetLoader()
    (x_train, y_train), (x_test, y_test), class_names = loader.load_cifar10()

    input_shape, _ = get_data_information(x_train, y_train)

    model = DCGANCifar10(
        input_shape=input_shape,
        noise_dim=110,
        fake_activation='tanh',
        batch_size=64,
        optimizer='adam',
        learning_rate=0.0002,
        adam_beta_1=0.5,
        epochs=15,
        n_fid_samples=5000,
        tf_verbose=False
    )
    model.fit(x_train, log_dir='log_dcgan-cifar10', log_period=1)
    model.predict(plot=True)
