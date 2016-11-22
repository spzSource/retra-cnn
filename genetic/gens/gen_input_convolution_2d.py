from keras.layers import Convolution2D

from genetic.gen_type import EncodedType
from genetic.gens.gen_convolution_2d import Convolution2DGen


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