from keras.layers import Convolution2D

from genetic.gen_type import GenType
from genetic.gens.gen_convolution_2d import Convolution2DGen


class InputConvolution2DGen(Convolution2DGen):
    """
    Represents gen for input convolution layer,
    which is based on regular gen for convolution layer.
    """
    def __init__(self, shape):
        super(InputConvolution2DGen, self).__init__()
        self.shape = shape

    @property
    def type(self):
        return GenType.InputConvolution2DGen

    def decode(self):
        return Convolution2D(
            self.filters_count,
            self.filter_size,
            self.filter_size,
            input_shape=self.shape)

