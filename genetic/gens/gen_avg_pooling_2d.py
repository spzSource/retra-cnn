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

    def encode(self, chromosome):
        result = list(chromosome)

        if GenType.Dense not in map(lambda gen: gen[0], result):
            self.size = random.randint(2, 8)
            result.append((self.type, [self.size]))
            result = ActivationGen().encode(result)

        return result

    def decode(self, encoded_gen):
        super(AvgPooling2DGen, self).decode(encoded_gen)

        (_, encoded_params) = encoded_gen

        return AveragePooling2D(
            strides=None,
            border_mode="same",
            pool_size=(encoded_params[0], encoded_params[0]))