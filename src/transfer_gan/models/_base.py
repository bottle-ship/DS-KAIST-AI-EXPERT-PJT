import tensorflow as tf

from abc import abstractmethod
from sklearn.base import BaseEstimator
from tensorflow.python.keras import backend


class BaseModel(BaseEstimator):

    def __init__(self):
        super(BaseModel, self).__init__()

        tf.enable_eager_execution()

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        backend.set_session(tf.compat.v1.Session(config=config))

    @abstractmethod
    def fit(self, x, y, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save_model(self, model_dir_name):
        raise NotImplementedError

    @abstractmethod
    def load_model(self, model_dir_name):
        raise NotImplementedError
