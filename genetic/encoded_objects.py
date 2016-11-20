import random

from aetypes import Enum
from abc import abstractproperty, abstractmethod

from keras.layers import Dense, Activation, AveragePooling2D, Convolution2D


class EncodedType(Enum):
    Convolution2d = 1
    Activation = 2
    Dense = 3
    AvgPooling2d = 4


class GenObject(object):
    """
    An abstract gen which is combined to chromosome.
    :type - property is used for encoding and decoding layer object.
    """
    @abstractproperty
    def type(self):
        pass

    @abstractmethod
    def encode(self, chromosome):
        pass

    def decode(self, encoded_gen):
        (gen_type, _) = encoded_gen
        if gen_type == self.type:
            raise Exception("Wrong encoded data type.")


class DenseGen(GenObject):
    def __init__(self, size=0):
        self.size = size

    @property
    def type(self):
        return EncodedType.Dense

    def encode(self, chromosome):
        result = list(chromosome)

        self.size = random.randint(32, 4096)
        result.append((self.type, [self.size]))
        result.append(EncodedType.Activation)

        return result

    def decode(self, encoded_gen):
        super(DenseGen, self).decode(encoded_gen)

        (_, encoded_params) = encoded_gen
        return Dense(output_dim=encoded_params[0])


class ActivationGen(GenObject):
    def __init__(self):
        pass

    @property
    def type(self):
        return EncodedType.Activation

    def encode(self, chromosome):
        result = list(chromosome)
        if len(result) > 0 and not result[-1] == EncodedType.Activation:
            result.append(self.type)
        return result

    def decode(self, encoded_gen):
        super(ActivationGen, self).decode(encoded_gen)
        return Activation(activation='relu')


class AvgPooling2DGen(GenObject):
    def __init__(self, size=0):
        self.size = size

    @property
    def type(self):
        return EncodedType.AvgPooling2d

    def encode(self, chromosome):
        result = list(chromosome)

        self.size = random.randint(2, 8)
        result.append((self.type, [self.size]))
        result.append(EncodedType.Activation)

        return result

    def decode(self, encoded_gen):
        super(AvgPooling2DGen, self).decode(encoded_gen)

        (_, encoded_params) = encoded_gen

        return AveragePooling2D(
            strides=None,
            border_mode="same",
            pool_size=(encoded_gen[0], encoded_gen[0]))


class Convolution2DGen(GenObject):
    def __init__(self, filters_count=0, filter_size=0):
        self.filter_size = filter_size
        self.filters_count = filters_count

    @property
    def type(self):
        return EncodedType.Convolution2d

    def encode(self, chromosome):
        result = list(chromosome)

        self.filter_size = random.randint(2, 8)
        self.filters_count = random.randint(2, 16)

        result.append((self.type, [self.filters_count, self.filter_size]))
        result.append(EncodedType.Activation)

        return result

    def decode(self, encoded_gen):
        super(Convolution2DGen, self).decode(encoded_gen)

        (_, encoded_params) = encoded_gen

        return Convolution2D(encoded_gen[0], encoded_gen[1], encoded_gen[1], activation="relu")

