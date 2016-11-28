from keras.layers import Activation

from genetic.gens.gen import Gen
from genetic.gen_type import GenType


class ActivationGen(Gen):
    """
    Activation layer's gen representation.
    """
    def __init__(self):
        pass

    @property
    def type(self):
        return GenType.Activation

    def encode(self):
        return self.type, []

    def decode(self, encoded_gen):
        super(ActivationGen, self).decode(encoded_gen)
        return Activation(activation='relu')


class OutputActivation(Gen):
    @property
    def type(self):
        return GenType.OutputActivation

    def encode(self):
        return self.type, []

    def decode(self, encoded_gen):
        super(OutputActivation, self).decode(encoded_gen)
        return Activation(activation='softmax')
