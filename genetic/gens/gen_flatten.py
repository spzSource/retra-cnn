from keras.layers import Flatten

from genetic.gens.gen_object import GenObject
from genetic.gen_type import EncodedType


class FlattenGen(GenObject):
    """
    Represents flatten layer,
    which is converts n-d layer to (n-1)-d layer.
    """
    @property
    def type(self):
        return EncodedType.Flatten

    def encode(self, chromosome):
        result = list(chromosome)
        result.append((EncodedType.Flatten, []))
        return result

    def decode(self, encoded_gen):
        super(FlattenGen, self).decode(encoded_gen)
        return Flatten()