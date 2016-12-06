from keras.layers import Dense

from genetic.gen_type import GenType
from genetic.gens.gen import Gen
from genetic.gens.gen_dense import DenseGen


class OutputDenseGen(Gen):
    """
    Represents gen for output layer for neural network.
    """
    def __init__(self, size=10):
        self.size = size

    def __str__(self):
        return "Dense: size={0}".format(self.size)

    @property
    def type(self):
        return GenType.OutputDense

    def decode(self):
        return Dense(self.size, activation="softmax", init="lecun_uniform")
