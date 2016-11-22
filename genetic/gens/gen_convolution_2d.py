import random

from keras.layers import Convolution2D

from genetic.gens.gen import Gen
from genetic.gen_type import EncodedType
from genetic.gens.gen_activation import ActivationGen


class Convolution2DGen(Gen):
    """
    Convolution later's gen representation.
    :param filters_count - the number of feature-maps.
    :param filter_size - the size of feature-map (NxM).
    """
    def __init__(self, filters_count=0, filter_size=0):
        self.filter_size = filter_size
        self.filters_count = filters_count

    @property
    def type(self):
        return EncodedType.Convolution2d

    def encode(self, chromosome):
        result = list(chromosome)

        if EncodedType.Dense not in map(lambda gen: gen[0], result):
            self.filter_size = random.randint(2, 8)
            self.filters_count = random.randint(2, 16)

            result.append((self.type, [self.filters_count, self.filter_size]))
            result = ActivationGen().encode(result)

        return result

    def decode(self, encoded_gen):
        super(Convolution2DGen, self).decode(encoded_gen)

        (_, encoded_params) = encoded_gen

        return Convolution2D(encoded_params[0], encoded_params[1], encoded_params[1])