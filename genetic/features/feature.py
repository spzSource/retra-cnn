from abc import abstractmethod


class Feature(object):
    """
    A piece of logic which should be applied against target chromosome.
    """
    @abstractmethod
    def evaluate(self, chromosome):
        pass
