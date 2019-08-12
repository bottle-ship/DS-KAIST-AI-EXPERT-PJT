from transfer_gan.datasets import DatasetLoader
from transfer_gan.utils.visualization import show_image_from_dataset


if __name__ == '__main__':
    loader = DatasetLoader()
    (x_train, y_train), (x_test, y_test), class_names = loader.load_tiny_imagenet()

    print(x_train.shape)
    print(y_train.shape)
    print(class_names)

    show_image_from_dataset(x_train, y_train, class_names)
