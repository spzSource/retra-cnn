from abc import abstractmethod


class ChromosomeEstimation(object):
    """
    A contract for estimation metric of neural network.
    """

    @abstractmethod
    def estimate(self, chromosome):
        """
        Performs estimation for target chromosome.
        :param chromosome: the chromosome to be estimated.
        :return: value of estimated value.s
        """
        pass
