import random

from keras.layers import Dense

from genetic.gen_type import GenType
from genetic.gens.gen import Gen


class DenseGen(Gen):
    """
    Fully connected layers's gen representation
    """
    def __init__(self):
        self.size = random.randint(32, 2048)

    def __str__(self):
        return "Dense: size={0}".format(self.size)

    @property
    def type(self):
        return GenType.Dense

    def decode(self):
        return Dense(self.size, init="lecun_uniform")
