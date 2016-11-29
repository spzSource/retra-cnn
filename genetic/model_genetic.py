import random

import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.optimizers import SGD
from pyeasyga.pyeasyga import GeneticAlgorithm

from genetic.chromosome import Chromosome
from genetic.gen_type import GenType
from genetic.gens.gen_activation import ActivationGen, OutputActivation
from genetic.gens.gen_convolution_2d import Convolution2DGen
from genetic.gens.gen_dense import DenseGen
from genetic.gens.gen_flatten import FlattenGen
from genetic.gens.gen_input_convolution_2d import InputConvolution2DGen
from genetic.gens.gen_output_dense import OutputDenseGen
from genetic.strategies.activation_attach_strategy import ActivationAttachStrategy
from genetic.strategies.convolution2d_attach_strategy import Convolution2dAttachStrategy
from genetic.strategies.dense_attach_strategy import DenseAttachStrategy
from genetic.strategies.input_attach_strategy import InputConvolution2dAttachStrategy
from genetic.strategies.output_dense_attach_strategy import OutputDenseAttachStrategy


class GeneticClassificationModel(object):
    """
    Performs evolution model for classification.
    First step is creation a some set of initial neural networks.
    Gen structure is array of neurons for each or layers in the target neural network.
    """

    def __init__(self):

        def _to_expected_output(expected_output):
            class_number = expected_output[0]
            result = np.zeros(10, dtype=np.int)
            result[class_number] = 1
            return result

        self.attach_strategies = [
            DenseAttachStrategy(),
            ActivationAttachStrategy(),
            Convolution2dAttachStrategy(),
            OutputDenseAttachStrategy(),
            InputConvolution2dAttachStrategy()
        ]

        self.encoding_map = {
            GenType.Dense: DenseGen(),
            GenType.Flatten: FlattenGen(),
            GenType.Activation: ActivationGen(),
            GenType.Convolution2d: Convolution2DGen(),
            GenType.OutputDense: OutputDenseGen(),
            GenType.InputConvolution2DGen: InputConvolution2DGen(),
            GenType.OutputActivation: OutputActivation()
        }

        self.genetic = GeneticAlgorithm([], population_size=10)
        self.genetic.mutate_function = self._mutation
        self.genetic.fitness_function = self._fitness
        self.genetic.crossover_function = self._crossover
        self.genetic.create_individual = self._create_individual

        self.training_set, _ = cifar10.load_data()

        (inputs, expected_outputs) = self.training_set
        self.np_input = np.array(inputs[:3])
        self.np_expected = np.array(map(_to_expected_output, expected_outputs)[:3])

    def fit(self):
        self.genetic.run()

    def _fitness(self, member, _):
        """
        Calculates score for each of members of current population.
        :param member: the member of current population.
        :return: score value.
        """

        internal_layers = self._decode_chromosome(member)

        try:
            model = Sequential(internal_layers)
            model.compile(
                loss="mse",
                metrics=["accuracy"],
                optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False))

            history = model.fit(self.np_input, self.np_expected, batch_size=10, nb_epoch=10)
            ratio = history.history["acc"][-1]
        except ValueError as e:
            print(e)
            ratio = 0

        return ratio

    def _crossover(self, parent1, parent2):
        """
        Performs crossover operator for two parents.
        Operator produces two children with genotype of both parents.
        :param parent1: the first parent.
        :param parent2: the second parent.
        :return: two chromosomes which is a children of two parents.
        """
        return parent1.cross(parent2)

    def _mutation(self, chromosome):
        """
        Performs mutation operator for specified chromosome.
        :param chromosome: the set of gens (chromosome).
        :return: the index of added gen
        """
        return chromosome.mutate()

    def _create_individual(self, _):
        """
        Randomly creates chromosome using special encoding map.
        :return: array of gens - chromosome.
        """
        chromosome = Chromosome()

        auto_generated_types = [
            GenType.Convolution2d,
            GenType.Dense,
            GenType.Activation
        ]

        gen_seq = [GenType.InputConvolution2DGen, GenType.Activation]
        seq = [random.choice(auto_generated_types) for _ in range(0, random.randint(2, 16))]
        gen_seq += seq
        gen_seq += [GenType.OutputDense, GenType.OutputActivation]

        for layer_type in gen_seq:
            for strategy in self.attach_strategies:
                if strategy.target_type == layer_type:
                    strategy.evaluate(chromosome, self.encoding_map[layer_type])
                    break

        return chromosome

    def _decode_chromosome(self, chromosome):
        """
        Converts encoded chromosome to array of layers.
        :param chromosome: array of encoded gens.
        :return: array of layers for neural network.
        """
        layers = []
        for gen in chromosome.gens:
            layer = self.encoding_map[gen.type].decode(gen.encode())
            layers.append(layer)
        return layers
