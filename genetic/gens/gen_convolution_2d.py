import random

from keras.layers import Convolution2D

from genetic.gens.gen import Gen
from genetic.gen_type import GenType
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
        return GenType.Convolution2d

    def encode(self):
        self.filter_size = random.randint(2, 8)
        self.filters_count = random.randint(2, 16)
        return self.type, [self.filters_count, self.filter_size]

    def decode(self, encoded_gen):
        super(Convolution2DGen, self).decode(encoded_gen)

        (_, encoded_params) = encoded_gen

        return Convolution2D(encoded_params[0], encoded_params[1], encoded_params[1])