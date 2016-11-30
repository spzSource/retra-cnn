from keras.layers import Activation

from genetic.gens.gen import Gen
from genetic.gen_type import GenType


class ActivationGen(Gen):
    """
    Activation layer's gen representation.
    """
    def __init__(self):
        pass

    def __str__(self):
        return "Activation(relu)"

    @property
    def type(self):
        return GenType.Activation

    def decode(self):
        return Activation(activation='relu')


class OutputActivation(Gen):

    def __str__(self):
        return "Activation(softmax)"

    @property
    def type(self):
        return GenType.OutputActivation

    def decode(self):
        return Activation(activation='softmax')
