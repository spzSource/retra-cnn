from abc import abstractproperty, abstractmethod


class Gen(object):
    """
    An abstract gen which is combined to chromosome.
    type - property is used for encoding and decoding layer object.
    """
    @abstractproperty
    def type(self):
        pass

    @abstractmethod
    def decode(self):
        pass
