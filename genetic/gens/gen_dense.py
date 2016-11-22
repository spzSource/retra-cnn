import random

from keras.layers import Dense

from genetic.gens.gen_object import GenObject
from genetic.gen_type import EncodedType
from genetic.gens.gen_flatten import FlattenGen
from genetic.gens.gen_activation import ActivationGen


class DenseGen(GenObject):
    """
    Fully connected layers's gen representation
    :param size - the number of neurons for current layer.
    """
    def __init__(self, size=0):
        self.size = size

    @property
    def type(self):
        return EncodedType.Dense

    def encode(self, chromosome):
        result = list(chromosome)
        if len(result) > 0:
            (last_gen_type, _) = result[-1]
            if last_gen_type == EncodedType.Activation and len(result) > 1:
                (last_gen_type, _) = result[-2]
                if last_gen_type in [EncodedType.AvgPooling2d, EncodedType.Convolution2d,
                                     EncodedType.InputConvolution2DGen]:
                    result = FlattenGen().encode(result)
            elif last_gen_type in [EncodedType.AvgPooling2d, EncodedType.Convolution2d,
                                   EncodedType.InputConvolution2DGen]:
                result = FlattenGen().encode(result)

        self.size = random.randint(32, 4096)
        result.append((self.type, [self.size]))
        result = ActivationGen().encode(result)

        return result

    def decode(self, encoded_gen):
        super(DenseGen, self).decode(encoded_gen)

        (_, encoded_params) = encoded_gen

        return Dense(encoded_params[0])