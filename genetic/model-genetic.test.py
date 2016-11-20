import random
import unittest
from aetypes import Enum

from pyeasyga.pyeasyga import GeneticAlgorithm
from abc import abstractproperty, abstractmethod


class EncodedType(Enum):
    Convolution2d = 1
    Activation = 2
    Dense = 3
    AvgPooling2d = 4


class EncodedObject(object):
    """
    Encoded object type.
    Type property is used for encoding and decoding layer object.
    """
    @abstractproperty
    def type(self):
        pass

    @abstractmethod
    def encode(self, chromosome):
        pass


class DenseEncoded(EncodedObject):
    def __init__(self, size=0):
        self.size = size

    @property
    def type(self):
        return EncodedType.Dense

    def encode(self, chromosome):
        result = list(chromosome)

        layer_size = random.randint(32, 4096)
        result.append((self.type, [layer_size]))
        result.append(2)

        return result


class ActivationEncoded(EncodedObject):
    def __init__(self):
        pass

    @property
    def type(self):
        return EncodedType.Activation

    def encode(self, chromosome):
        result = list(chromosome)
        if len(result) > 0 and not result[-1] == 2:
            result.append(self.type)
        return result


class AvgPooling2dEncoded(EncodedObject):
    def __init__(self, size=0):
        self.size = size

    @property
    def type(self):
        return EncodedType.AvgPooling2d

    def encode(self, chromosome):
        result = list(chromosome)

        pooling_size = random.randint(2, 8)
        result.append((self.type, [pooling_size]))
        result.append(2)

        return result


class Convolution2dEncoded(EncodedObject):
    def __init__(self, filters_count=0, filter_size=0):
        self.filter_size = filter_size
        self.filters_count = filters_count

    @property
    def type(self):
        return EncodedType.Convolution2d

    def encode(self, chromosome):
        result = list(chromosome)

        filter_size = random.randint(2, 8)
        filters_count = random.randint(2, 16)

        result.append((self.type, [filters_count, filter_size]))
        result.append(2)

        return result


class GeneticClassificationModelTest(unittest.TestCase):
    """
    Layers encoding strategy:
        1 - conv2d (n_conv, (x, y))
        2 - activation
        3 - dense (size)
        4 - avg2d ((x, y))

    Example of chromosome:
        static: conv2d(16, (8, 8), inpt=(3, 32, 32))
        static: activation(relu)
        ---
        conv2d(8, (4, 4))  | 1: (8, (4, 4))
        activation(relu)   | 2
        avg2d(2, 2)        | 4: (2, 2)
        dense(256)         | 3: (256)
        activation(relu)   | 2
        ---
        static: dense(10)
        static: activation(softmax)

        chromosome: { 1: (8, (4, 4)) }---{ 2 }---{ 4: (2, 2) }---{ 3: (256) }---{ 2 }

    """

    @staticmethod
    def fitness(member, data):
        pass

    @staticmethod
    def crossover(parent1, parent2):
        pass

    @staticmethod
    def create_individual(encoding_map):
        chromosome = []

        for layer_index in range(0, random.randint(2, 16)):
            layer_type = random.randint(1, 4)
            chromosome = encoding_map[layer_type].encode(chromosome)

        if len(chromosome) > 0 and not chromosome[-1] == 2:
            chromosome.append(2)
        return chromosome

    @staticmethod
    def mutation(individual):
        pass

    def setUp(self):

        """Genetic algorithm initialization"""
        self.encoded_map = {
            EncodedType.Dense: DenseEncoded(),
            EncodedType.Activation: ActivationEncoded(),
            EncodedType.AvgPooling2d: AvgPooling2dEncoded(),
            EncodedType.Convolution2d: Convolution2dEncoded()
        }

        genetic = GeneticAlgorithm(list(self.encoded_map))
        genetic.mutate_function = self.mutation
        genetic.fitness_function = self.fitness
        genetic.create_individual = self.create_individual
        genetic.crossover_function = self.crossover

        # genetic.run()

    def test_create_individual(self):

        """Test consistency of individual"""
        individual = self.create_individual(self.encoded_map)

        assert individual
        assert EncodedType.Activation in individual
        assert not EncodedType.Activation == individual[0]
        assert individual[1::2].count(EncodedType.Activation) == len(individual[1::2])


if __name__ == '__main__':
    unittest.main()
