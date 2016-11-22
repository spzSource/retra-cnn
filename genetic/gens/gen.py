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
    def encode(self, chromosome):
        pass

    def decode(self, encoded_gen):
        (gen_type, _) = encoded_gen
        if not gen_type == self.type:
            raise Exception("Wrong encoded data type. Expected {0}, but actual is {1}".format(self.type, gen_type))