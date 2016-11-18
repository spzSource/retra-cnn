import numpy as np

from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D, AveragePooling2D


class ClassificationModel(object):
    """
    Classification classifier.
    It is a convolutional neural network which is recieve image on input
    and returns the array [0..9] with possibilities of belonging for such classes.
    """

    def __init__(self):
        self.model = None

    def build(self):
        """
        Builds classification classifier.
        """
        self.model = Sequential([
            Convolution2D(8, 5, 5, input_shape=(3, 32, 32), activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'),
            Convolution2D(16, 3, 3, activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'),
            Convolution2D(32, 2, 2, activation='relu'),
            AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='same'),
            # MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'),
            Flatten(),
            Dense(10, activation='softmax')
        ])

        self.model.compile(
            loss='mse',
            metrics=['accuracy', 'mean_absolute_percentage_error'],
            optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False))

    def train(self, training_set):
        (inputs, expected_outputs) = training_set
        self.model.fit(np.array(inputs), np.array(map(self._to_expected_output, expected_outputs)), nb_epoch=1000, batch_size=32)

    @staticmethod
    def _to_expected_output(expected_output):
        class_number = expected_output[0]
        result = np.zeros(10, dtype=np.int)
        result[class_number] = 1
        return result
