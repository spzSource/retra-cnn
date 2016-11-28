from abc import abstractmethod, abstractproperty


class Constraint(object):
    """
    A piece of logic which should be applied against target chromosome.
    """

    @abstractproperty
    def target_types(self):
        pass

    @abstractmethod
    def evaluate(self, chromosome):
        pass
