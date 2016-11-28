import random

from keras.layers import AveragePooling2D

from genetic.gens.gen import Gen
from genetic.gen_type import GenType
from genetic.gens.gen_activation import ActivationGen


class AvgPooling2DGen(Gen):
    """
    Average Pooling layer's gen representation.
    :param size - the size of pooling window.
    """
    def __init__(self, size=0):
        self.size = size

    @property
    def type(self):
        return GenType.AvgPooling2d

    def encode(self):
        self.size = random.randint(2, 8)
        return self.type, [self.size]

    def decode(self, encoded_gen):
        super(AvgPooling2DGen, self).decode(encoded_gen)

        (_, encoded_params) = encoded_gen

        return AveragePooling2D(
            strides=None,
            border_mode="same",
            pool_size=(encoded_params[0], encoded_params[0]))