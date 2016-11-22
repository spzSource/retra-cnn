import random

from aetypes import Enum
from abc import abstractproperty, abstractmethod

from keras.layers import Dense, Activation, AveragePooling2D, Convolution2D, Flatten


class EncodedType(Enum):
    Convolution2d = 1
    Activation = 2
    Dense = 3
    AvgPooling2d = 4
    Flatten = 5
    InputConvolution2DGen = 6
    OutputDense = 7


class GenObject(object):
    """
    An abstract gen which is combined to chromosome.
    type - property is used for encoding and decoding layer object.
    """
    @abstractproperty
    def type(self):
        pass

    @abstractmethod
    def encode(self, chromosome):
        pass

    def decode(self, encoded_gen):
        (gen_type, _) = encoded_gen
        if not gen_type == self.type:
            raise Exception("Wrong encoded data type. Expected {0}, but actual is {1}".format(self.type, gen_type))


class DenseGen(GenObject):
    """
    Fully connected layers's gen representation
    :param size - the number of neurons for current layer.
    """
    def __init__(self, size=0):
        self.size = size

    @property
    def type(self):
        return EncodedType.Dense

    def encode(self, chromosome):
        result = list(chromosome)
        if len(result) > 0:
            (last_gen_type, _) = result[-1]
            if last_gen_type == EncodedType.Activation and len(result) > 1:
                (last_gen_type, _) = result[-2]
                if last_gen_type in [EncodedType.AvgPooling2d, EncodedType.Convolution2d,
                                     EncodedType.InputConvolution2DGen]:
                    result = FlattenGen().encode(result)
            elif last_gen_type in [EncodedType.AvgPooling2d, EncodedType.Convolution2d,
                                   EncodedType.InputConvolution2DGen]:
                result = FlattenGen().encode(result)

        self.size = random.randint(32, 4096)
        result.append((self.type, [self.size]))
        result = ActivationGen().encode(result)

        return result

    def decode(self, encoded_gen):
        super(DenseGen, self).decode(encoded_gen)

        (_, encoded_params) = encoded_gen

        return Dense(encoded_params[0])


class ActivationGen(GenObject):
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


class AvgPooling2DGen(GenObject):
    """
    Average Pooling layer's gen representation.
    :param size - the size of pooling window.
    """
    def __init__(self, size=0):
        self.size = size

    @property
    def type(self):
        return EncodedType.AvgPooling2d

    def encode(self, chromosome):
        result = list(chromosome)

        if EncodedType.Dense not in map(lambda gen: gen[0], result):
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


class Convolution2DGen(GenObject):
    """
    Convolution later's gen representation.
    :param filters_count - the number of feature-maps.
    :param filter_size - the size of feature-map (NxM).
    """
    def __init__(self, filters_count=0, filter_size=0):
        self.filter_size = filter_size
        self.filters_count = filters_count

    @property
    def type(self):
        return EncodedType.Convolution2d

    def encode(self, chromosome):
        result = list(chromosome)

        if EncodedType.Dense not in map(lambda gen: gen[0], result):
            self.filter_size = random.randint(2, 8)
            self.filters_count = random.randint(2, 16)

            result.append((self.type, [self.filters_count, self.filter_size]))
            result = ActivationGen().encode(result)

        return result

    def decode(self, encoded_gen):
        super(Convolution2DGen, self).decode(encoded_gen)

        (_, encoded_params) = encoded_gen

        return Convolution2D(encoded_params[0], encoded_params[1], encoded_params[1])


class InputConvolution2DGen(Convolution2DGen):
    """
    Represents gen for input convolution layer,
    which is based on regular gen for convolution layer.
    """
    @property
    def type(self):
        return EncodedType.InputConvolution2DGen

    def decode(self, encoded_gen):
        super(Convolution2DGen, self).decode(encoded_gen)

        (_, encoded_params) = encoded_gen
        return Convolution2D(encoded_params[0], encoded_params[1], encoded_params[1], input_shape=(3, 32, 32))


class OutputDenseGen(DenseGen):
    """
    Represents gen for output layer for neural network.
    """
    @property
    def type(self):
        return EncodedType.OutputDense

    def decode(self, encoded_gen):
        super(DenseGen, self).decode(encoded_gen)
        return Dense(10)


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
