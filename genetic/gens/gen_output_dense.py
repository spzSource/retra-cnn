from keras.layers import Dense

from genetic.gen_type import GenType
from genetic.gens.gen_dense import DenseGen


class OutputDenseGen(DenseGen):
    """
    Represents gen for output layer for neural network.
    """
    @property
    def type(self):
        return GenType.OutputDense

    def decode(self, encoded_gen):
        super(DenseGen, self).decode(encoded_gen)
        return Dense(10, activation="softmax")