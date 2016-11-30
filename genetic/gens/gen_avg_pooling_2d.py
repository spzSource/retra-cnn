import random

from keras.layers import AveragePooling2D

from genetic.gens.gen import Gen
from genetic.gen_type import GenType
from genetic.gens.gen_activation import ActivationGen


class AvgPooling2DGen(Gen):
    """
    Average Pooling layer's gen representation.
    """
    def __init__(self):
        self.size = random.randint(2, 8)

    def __str__(self):
        return "AvgPooling2D: pool_size=({0}, {0}), border_mode=same".format(self.size)

    @property
    def type(self):
        return GenType.AvgPooling2d

    def decode(self):
        return AveragePooling2D(
            strides=None,
            border_mode="same",
            pool_size=(self.size, self.size))
