import unittest

from genetic.encoded_objects import *


class GeneticClassificationModelTest(unittest.TestCase):

    """
    Layers encoding strategy:
        1 - conv2d (n_conv, (x, y))
        2 - activation
        3 - dense (size)
        4 - avg2d ((x, y))
    """

    def setUp(self):
        pass

    def test_create_individual(self):

        """Test consistency of individual"""
        individual = self.create_individual(self.encoded_map)

        assert individual
        assert EncodedType.Activation in individual
        assert not EncodedType.Activation == individual[0]
        assert individual[1::2].count(EncodedType.Activation) == len(individual[1::2])


if __name__ == '__main__':
    unittest.main()
