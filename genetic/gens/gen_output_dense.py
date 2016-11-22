from keras.layers import Dense

from genetic.gen_type import EncodedType
from genetic.gens.gen_dense import DenseGen


class OutputDenseGen(DenseGen):
    """
    Represents gen for output layer for neural network.
    """
    @property
    def type(self):
        return EncodedType.OutputDense

    def decode(self, encoded_gen):
        super(DenseGen, self).decode(encoded_gen)
        return Dense(10)