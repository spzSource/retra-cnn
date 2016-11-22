from aetypes import Enum


class EncodedType(Enum):
    Convolution2d = 1
    Activation = 2
    Dense = 3
    AvgPooling2d = 4
    Flatten = 5
    InputConvolution2DGen = 6
    OutputDense = 7