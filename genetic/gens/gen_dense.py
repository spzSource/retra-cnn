import random

from keras.layers import Dense

from genetic.constraints.constraint_activation_after import ActivationAfterConstraint
from genetic.constraints.constraint_flatten_before import FlattenBeforeConstraint
from genetic.gens.gen import Gen
from genetic.gen_type import GenType
from genetic.gens.gen_flatten import FlattenGen
from genetic.gens.gen_activation import ActivationGen


class DenseGen(Gen):
    """
    Fully connected layers's gen representation
    :param size - the number of neurons for current layer.
    """
    def __init__(self, size=0):
        self.size = size

    @property
    def type(self):
        return GenType.Dense

    def encode(self):
        self.size = random.randint(32, 4096)
        return self.type, [self.size]

    def decode(self, encoded_gen):
        super(DenseGen, self).decode(encoded_gen)

        (_, encoded_params) = encoded_gen

        return Dense(encoded_params[0])