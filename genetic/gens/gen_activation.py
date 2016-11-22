from keras.layers import Activation

from genetic.gens.gen import Gen
from genetic.gen_type import EncodedType


class ActivationGen(Gen):
    """
    Activation layer's gen representation.
    """
    def __init__(self):
        pass

    @property
    def type(self):
        return EncodedType.Activation

    def encode(self, chromosome):
        result = list(chromosome)
        if len(result) > 0 and not result[-1][0] == EncodedType.Activation:
            result.append((self.type, []))
        return result

    def decode(self, encoded_gen):
        super(ActivationGen, self).decode(encoded_gen)
        return Activation(activation='relu')