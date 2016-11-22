import random

from keras.layers import Dense

from genetic.features.feature_activation_after import ActivationAfterFeature
from genetic.features.feature_flatten_before import FlattenBeforeFeature
from genetic.gens.gen import Gen
from genetic.gen_type import GenType
from genetic.gens.gen_flatten import FlattenGen
from genetic.gens.gen_activation import ActivationGen


class DenseGen(Gen):
    """
    Fully connected layers's gen representation
    :param size - the number of neurons for current layer.
    """
    def __init__(self, size=0, features=None):

        if features is None:
            features = []

        self.size = size
        self.features = features

    @property
    def type(self):
        return GenType.Dense

    def encode(self, chromosome):
        result = list(chromosome)

        for feature in self.features:
            result = feature.evaluate(result)

        self.size = random.randint(32, 4096)
        result.append((self.type, [self.size]))

        return result

    def decode(self, encoded_gen):
        super(DenseGen, self).decode(encoded_gen)

        (_, encoded_params) = encoded_gen

        return Dense(encoded_params[0])