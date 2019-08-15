from transfer_gan.datasets import DatasetLoader
from transfer_gan.models.lsgan import LSGANMnist
from transfer_gan.utils.data_utils import get_data_information


if __name__ == '__main__':
    loader = DatasetLoader()
    (x_train, y_train), (x_test, y_test), class_names = loader.load_mnist()

    input_shape, _ = get_data_information(x_train, y_train)

    model = LSGANMnist(
        input_shape=input_shape,
        noise_dim=100,
        fake_activation='tanh',
        batch_size=32,
        epochs=50,
        tf_verbose=False
    )
    model.fit(x_train, log_dir='log_lcgan-mnist', log_period=1)
    model.predict(plot=True)
