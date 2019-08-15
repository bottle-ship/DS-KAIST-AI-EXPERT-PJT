import os

import tensorflow as tf

from abc import abstractmethod
from sklearn.base import BaseEstimator
from tensorflow.python.keras import backend


class BaseModel(BaseEstimator):

    def __init__(self, tf_verbose):
        super(BaseModel, self).__init__()

        if not tf_verbose:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        tf.compat.v1.enable_eager_execution()
        self.config = tf.compat.v1.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self._session = None
        self._create_session()

    def _create_session(self):
        self._session = tf.compat.v1.Session(config=self.config)
        backend.set_session(self._session)

    def _delete_session(self):
        backend.clear_session()
        self._session.__del__()
        self._session = None

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
