from keras.layers import Activation, Flatten
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD

from classifier import Classification
from classifier.model_visualized import VisualizedModel


class BestClassificationModel(Classification):
    def __init__(self):
        self._model = None

    @property
    def model(self):
        return self._model

    def build(self):
        self._model = Sequential([
            Convolution2D(6, 6, 6, input_shape=(3, 32, 32)),
            Activation(activation="relu"),
            Convolution2D(8, 8, 8),
            Activation(activation="relu"),
            Flatten(),
            Dense(1226),
            Activation(activation="relu"),
            Dense(790),
            Activation(activation="relu"),
            Dense(10),
            Activation(activation="softmax")
        ])
        self._model.compile(
            loss="mse",
            metrics=["accuracy"],
            optimizer=SGD(lr=0.03, momentum=0.0, decay=0.0, nesterov=False))

    def train(self, training_set):
        pass


if __name__ == "__main__":

    import sys

    model = VisualizedModel(origin=BestClassificationModel())
    model.build().show(sys.argv[1])
