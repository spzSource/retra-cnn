from keras.layers import Flatten

from genetic.gens.gen import Gen
from genetic.gen_type import GenType


class FlattenGen(Gen):
    """
    Represents flatten layer,
    which is converts n-d layer to (n-1)-d layer.
    """
    @property
    def type(self):
        return GenType.Flatten

    def encode(self):
        return GenType.Flatten, []

    def decode(self, encoded_gen):
        super(FlattenGen, self).decode(encoded_gen)
        return Flatten()