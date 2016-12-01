from abc import abstractmethod


class MutationStrategy(object):
    """
    Holds mutation login for target chromosomes.
    """
    @abstractmethod
    def check(self, chromosome):
        pass

    @abstractmethod
    def evaluate(self, chromosome):
        pass
