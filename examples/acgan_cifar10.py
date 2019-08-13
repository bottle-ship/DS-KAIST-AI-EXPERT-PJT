from transfer_gan.datasets import DatasetLoader
from transfer_gan.models.acgan import ACGANCifar10
from transfer_gan.utils.data_utils import get_data_information


if __name__ == '__main__':
    loader = DatasetLoader()
    (x_train, y_train), (x_test, y_test), class_names = loader.load_cifar10()

    input_shape, num_classes = get_data_information(x_train, y_train.ravel())

    model = ACGANCifar10(
        input_shape=input_shape,
        num_classes=num_classes,
        noise_dim=110,
        fake_activation='tanh',
        batch_size=64,
        learning_rate=0.0002,
        beta_1=0.5,
        epochs=15
    )
    model.fit(x_train, y_train.ravel(), log_dir='log_cifar10', log_period=1)
    model.predict(label=None, plot=True)
