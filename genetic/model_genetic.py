import random
import numpy as np

from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Convolution2D, Dense

from genetic.chromosome import Chromosome
from genetic.gen_type import GenType
from genetic.gens.gen_dense import DenseGen
from genetic.gens.gen_flatten import FlattenGen
from genetic.gens.gen_activation import ActivationGen, OutputActivation
from genetic.gens.gen_convolution_2d import Convolution2DGen
from genetic.constraints.constraint_flatten_before import FlattenBeforeConstraint
from genetic.constraints.constraint_activation_after import ActivationAfterConstraint

from pyeasyga.pyeasyga import GeneticAlgorithm

from genetic.gens.gen_input_convolution_2d import InputConvolution2DGen
from genetic.gens.gen_output_dense import OutputDenseGen


class GeneticClassificationModel(object):
    """
    Performs evolution model for classification.
    First step is creation a some set of initial neural networks.
    Gen structure is array of neurons for each or layers in the target neural network.
    """

    def __init__(self):

        self.features = [
            FlattenBeforeConstraint(),
            ActivationAfterConstraint()
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

    def fit(self):
        self.genetic.run()

    def _fitness(self, member, _):
        """
        Calculates score for each of members of current population.
        :param member: the member of current population.
        :return: score value.
        """

        def _to_expected_output(expected_output):
            class_number = expected_output[0]
            result = np.zeros(10, dtype=np.int)
            result[class_number] = 1
            return result

        input_layer = Convolution2D(8, 5, 5, input_shape=(3, 32, 32), activation="relu")
        output_layer = Dense(10, activation="softmax")
        internal_layers = self._decode_chromosome(member)

        print(map(lambda gen: gen.encode(), member.gens))
        layers = [input_layer] + internal_layers + [output_layer]
        print(layers)

        model = Sequential(layers)
        model.compile(
            loss="mse",
            metrics=["accuracy", "mean_absolute_percentage_error"],
            optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False))

        training_set, _ = cifar10.load_data()

        (inputs, expected_outputs) = training_set
        np_input = np.array(inputs[:10])
        np_expected = np.array(map(_to_expected_output, expected_outputs)[:10])

        try:
            history = model.fit(np_input, np_expected, batch_size=10, nb_epoch=100)
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

    def _create_individual(self, encoding_map):
        """
        Randomly creates chromosome using special encoding map.
        :return: array of gens - chromosome.
        """
        chromosome = Chromosome()
        chromosome = chromosome.push_back(InputConvolution2DGen())
        chromosome = chromosome.push_back(ActivationGen())

        for layer_index in range(0, random.randint(2, 16)):
            layer_type = random.randint(1, 3)

            if layer_type == GenType.Activation:
                if len(chromosome) > 0 and not chromosome.is_type_of(-1, GenType.Activation):
                    chromosome = chromosome.push_back(self.encoding_map[layer_type])

            elif layer_type == GenType.Convolution2d:
                if not chromosome.contains_type(GenType.Dense):
                    chromosome = chromosome.push_back(self.encoding_map[layer_type])
                    chromosome = chromosome.push_back(ActivationGen())

            elif layer_type == GenType.Dense:
                if not chromosome.contains_type(GenType.Dense):
                    chromosome = chromosome.push_back(FlattenGen())
                    chromosome = chromosome.push_back(self.encoding_map[layer_type])
                    chromosome = chromosome.push_back(ActivationGen())

            else:
                chromosome.push_back(self.encoding_map[layer_type])

        if not chromosome.contains_type(GenType.Dense):
            chromosome = chromosome.push_back(FlattenGen())
            chromosome = chromosome.push_back(OutputDenseGen())
        else:
            chromosome = chromosome.push_back(OutputDenseGen())

        if len(chromosome) > 0:
            chromosome = chromosome.push_back(OutputActivation())

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
