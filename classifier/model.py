from abc import abstractmethod, abstractproperty

import numpy as np

from keras.layers import \
    Dense, \
    Convolution2D, \
    Activation, GlobalAveragePooling2D

from keras.models import Sequential
from keras.optimizers import SGD


class Classification(object):

    @abstractproperty
    def model(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self, training_set):
        pass


class ClassificationModel(Classification):
    """
    Classification classifier.
    It is a convolution neural network which is receive image on input
    and returns the array [0..9] with possibilities of belonging for such classes.
    """

    def __init__(self):
        self._model = None

    @property
    def model(self):
        return self._model

    def build(self):
        """
        Builds classification classifier.
        """
        self._model = Sequential([
            Convolution2D(32, 16, 16, input_shape=(3, 32, 32)),
            Activation(activation="relu"),
            Convolution2D(16, 8, 8),
            Activation(activation="relu"),
            Convolution2D(8, 4, 4),
            Activation(activation="relu"),
            GlobalAveragePooling2D(),
            Activation(activation="relu"),
            Dense(10),
            Activation(activation="softmax")
        ])
        self._model.compile(
            loss="mse",
            metrics=["accuracy"],
            optimizer=SGD(lr=0.03, momentum=0.0, decay=0.0, nesterov=False))

    def train(self, training_set):
        """
        Performs model training using specified training set,
        :param training_set: a tuple (x_input, y_expected)
        :return: None
        """

        def _to_expected_output(expected_output):
            class_number = expected_output[0]
            result = np.zeros(10, dtype=np.int)
            result[class_number] = 1
            return result

        (inputs, expected_outputs) = training_set
        np_input = np.array(inputs)
        np_expected = np.array(map(_to_expected_output, expected_outputs))
        self._model.fit(np_input, np_expected, batch_size=32, nb_epoch=1000)
