import random
import unittest

from pyeasyga.pyeasyga import GeneticAlgorithm

class Conv2dEncodingData(object):
    def __init__(self, filters_count, filter_size):
        self.filter_size = filter_size
        self.filters_count = filters_count


class DenseEncodingData(object):
    def __init__(self, size):
        self.size = size


class Avg2dEncodingData(object):
    def __init__(self, size):
        self.size = size


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
    def create_individual(data):

        chromosome = []

        number_of_layers = random.randint(2, 16)
        for layer_index in range(0, number_of_layers):
            layer_type = random.randint(1, 4)

            if layer_type == 1:
                filter_size = random.randint(2, 8)
                filters_count = random.randint(2, 16)
                chromosome.append((layer_type, [filters_count, filter_size]))
                chromosome.append(2)

            if layer_type == 2:
                if len(chromosome) > 0 and not chromosome[-1] == 2:
                    chromosome.append(layer_type)

            if layer_type == 3:
                layer_size = random.randint(32, 4096)
                chromosome.append((layer_type, [layer_size]))
                chromosome.append(2)

            if layer_type == 4:
                pooling_size = random.randint(2, 8)
                chromosome.append((layer_type, [pooling_size]))
                chromosome.append(2)

        if not chromosome[-1] == 2:
            chromosome.append(2)
        return chromosome


    @staticmethod
    def mutation(individual):
        pass

    def setUp(self):

        """Genetic algorithm initialization"""
        genetic = GeneticAlgorithm([])
        genetic.mutate_function = self.mutation
        genetic.fitness_function = self.fitness
        genetic.create_individual = self.create_individual
        genetic.crossover_function = self.crossover

        # genetic.run()

    def test_create_individual(self):
        print(self.create_individual(None))


if __name__ == '__main__':
    unittest.main()





