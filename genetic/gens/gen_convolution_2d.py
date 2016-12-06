import random

from keras.layers import Convolution2D

from genetic.gens.gen import Gen
from genetic.gen_type import GenType
from genetic.gens.gen_activation import ActivationGen


class Convolution2DGen(Gen):
    """
    Convolution later's gen representation.
    """
    def __init__(self):
        self.filter_size = random.randint(2, 8)
        self.filters_count = random.randint(2, 8)

    def __str__(self):
        return "Convolution2D: n_filter={0}, nb_row={1}, nb_col={1}"\
            .format(self.filters_count, self.filters_count)

    @property
    def type(self):
        return GenType.Convolution2d

    def decode(self):
        return Convolution2D(self.filters_count, self.filter_size, self.filter_size, init="lecun_uniform")
