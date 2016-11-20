import random

from genetic.encoded_objects import \
    EncodedType, \
    DenseGen, \
    ActivationGen, \
    AvgPooling2DGen, \
    Convolution2DGen

from pyeasyga.pyeasyga import GeneticAlgorithm


class GeneticClassificationModel(object):
    """
    Performs evolution model for classification.
    First step is creation a some set of initial neural networks.
    Gen structure is array of neurons for each or layers in the target neural network.
    """
    def __init__(self):
        self.encoding_map = {
            EncodedType.Dense: DenseGen(),
            EncodedType.Activation: ActivationGen(),
            EncodedType.AvgPooling2d: AvgPooling2DGen(),
            EncodedType.Convolution2d: Convolution2DGen()
        }

        self.genetic = GeneticAlgorithm(self.encoding_map)
        self.genetic.mutate_function = self._mutation
        self.genetic.fitness_function = self._fitness
        self.genetic.crossover_function = self._crossover
        self.genetic.create_individual = self._create_individual

    def fit(self, training_data):
        pass

    @staticmethod
    def _fitness(member, encoding_map):
        """
        Calculates score for each of members of current population.
        :param member: the member of current population.
        :param encoding_map: the map which is used to convert chromosome to array of layers.
        :return: score value.
        """
        layers = GeneticClassificationModel._decode_chromosome(member, encoding_map)
        pass

    @staticmethod
    def _crossover(parent1, parent2):
        """
        Performs crossover operator for two parents.
        Operator produces two children with genotype of both parents.
        :param parent1: the first parent.
        :param parent2: the second parent.
        :return: two chromosomes which is a children of two parents.
        """
        if len(parent1) <= len(parent2):
            crossover_point = random.randrange(1, len(parent1))
        else:
            crossover_point = random.randrange(1, len(parent2))

        child_1 = parent1[:crossover_point] + parent2[crossover_point:]
        child_2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child_1, child_2

    @staticmethod
    def _mutation(individual):
        """
        Performs mutation operator for specified chromosome.
        :param individual: the set of gens (chromosome).
        :return: None
        """
        if len(individual) > 0:
            index = random.randrange(0, len(individual))
            individual.remove(index)

    @staticmethod
    def _create_individual(encoding_map):
        """
        Randomly creates chromosome using special encoding map.
        :return: array of gens - chromosome.
        """
        chromosome = []
        for layer_index in range(0, random.randint(2, 16)):
            layer_type = random.randint(1, 4)
            chromosome = encoding_map[layer_type].encode(chromosome)

        if len(chromosome) > 0 and not chromosome[-1] == EncodedType.Activation:
            chromosome.append(EncodedType.Activation)
        return chromosome

    @staticmethod
    def _decode_chromosome(chromosome, encoding_map):
        """
        Converts encoded chromosome to array of layers.
        :param chromosome: array of encoded gens.
        :param encoding_map: map which is helps to convert gen to layer object.
        :return: array of layers for neural network.
        """
        layers = []
        for gen in chromosome:
            (gen_type, gen_params) = gen
            layer = encoding_map[gen_type].decode(gen)
            layers.append(layer)
        return layers


