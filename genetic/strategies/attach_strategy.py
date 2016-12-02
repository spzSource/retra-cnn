from abc import abstractmethod, abstractproperty


class AttachStrategy(object):
    """
    A piece of logic which should be applied against target chromo.
    """

    @abstractproperty
    def target_type(self):
        pass

    @abstractmethod
    def evaluate(self, chromosome, gen):
        pass
