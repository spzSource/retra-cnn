from keras.layers import Dense

from genetic.gen_type import GenType
from genetic.gens.gen import Gen
from genetic.gens.gen_dense import DenseGen


class OutputDenseGen(Gen):
    """
    Represents gen for output layer for neural network.
    """
    @property
    def type(self):
        return GenType.OutputDense

    def encode(self):
        print(GenType.OutputDense, [])
        return GenType.OutputDense, []

    def decode(self, encoded_gen):
        super(OutputDenseGen, self).decode(encoded_gen)
        return Dense(10, activation="softmax")
