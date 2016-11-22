import random
import numpy as np

from keras.datasets import cifar10
from keras.layers import Convolution2D, Dense
from keras.models import Sequential
from keras.optimizers import SGD

from genetic.features.feature_activation_after import ActivationAfterFeature
from genetic.features.feature_flatten_before import FlattenBeforeFeature
from genetic.gen_type import GenType
from genetic.gens.gen_flatten import FlattenGen
from genetic.gens.gen_output_dense import OutputDenseGen
from genetic.gens.gen_input_convolution_2d import InputConvolution2DGen
from genetic.gens.gen_convolution_2d import Convolution2DGen
from genetic.gens.gen_activation import ActivationGen
from genetic.gens.gen_dense import DenseGen

from pyeasyga.pyeasyga import GeneticAlgorithm


class GeneticClassificationModel(object):
    """
    Performs evolution model for classification.
    First step is creation a some set of initial neural networks.
    Gen structure is array of neurons for each or layers in the target neural network.
    """

    def __init__(self):
        self.features = [
            FlattenBeforeFeature(),
            ActivationAfterFeature()
        ]

        self.encoding_map = {
            GenType.Dense: DenseGen(features=self.features),
            GenType.Flatten: FlattenGen(),
            GenType.Activation: ActivationGen(),
            GenType.OutputDense: OutputDenseGen(),
            # EncodedType.AvgPooling2d: AvgPooling2DGen(),
            GenType.Convolution2d: Convolution2DGen(features=[ActivationAfterFeature()]),
            GenType.InputConvolution2DGen: InputConvolution2DGen(features=[ActivationAfterFeature()])
        }

        self.genetic = GeneticAlgorithm(self.encoding_map)
        self.genetic.mutate_function = self._mutation
        self.genetic.fitness_function = self._fitness
        self.genetic.crossover_function = self._crossover
        self.genetic.create_individual = self._create_individual

    def fit(self):
        self.genetic.run()

    @staticmethod
    def _fitness(member, encoding_map):
        """
        Calculates score for each of members of current population.
        :param member: the member of current population.
        :param encoding_map: the map which is used to convert chromosome to array of layers.
        :return: score value.
        """
        internal_layers = GeneticClassificationModel._decode_chromosome(member, encoding_map)
        layers = [
                     Convolution2D(8, 5, 5, input_shape=(3, 32, 32), activation="relu"),
                 ] + internal_layers + [
                     Dense(10, activation="softmax")
                 ]
        print(map(lambda x: x.name, layers))
        model = Sequential(layers)

        model.compile(
            loss="mse",
            metrics=["accuracy", "mean_absolute_percentage_error"],
            optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False))

        training_set, _ = cifar10.load_data()

        (inputs, expected_outputs) = training_set
        np_input = np.array(inputs[:10])
        np_expected = np.array(map(GeneticClassificationModel._to_expected_output, expected_outputs)[:10])

        try:
            history = model.fit(np_input, np_expected, batch_size=10, nb_epoch=10)
            ratio = history.history["acc"][-1]
        except ValueError as e:
            print(e)
            ratio = 0

        return ratio

    @staticmethod
    def _to_expected_output(expected_output):
        class_number = expected_output[0]
        result = np.zeros(10, dtype=np.int)
        result[class_number] = 1
        return result

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
            individual.remove(individual[index])

    @staticmethod
    def _create_individual(encoding_map):
        """
        Randomly creates chromosome using special encoding map.
        :return: array of gens - chromosome.
        """
        chromosome = encoding_map[GenType.InputConvolution2DGen].encode([])

        for layer_index in range(0, random.randint(2, 16)):
            layer_type = random.randint(1, 4)
            if not layer_type == GenType.AvgPooling2d:
                chromosome = encoding_map[layer_type].encode(chromosome)

        layers2D = [
            GenType.AvgPooling2d,
            GenType.Convolution2d,
            GenType.InputConvolution2DGen
        ]

        if len(chromosome) > 0:
            (last_gen_type, _) = chromosome[-1]
            if last_gen_type == GenType.Activation and len(chromosome) > 1:
                (last_gen_type, _) = chromosome[-2]
                if last_gen_type in layers2D:
                    chromosome = FlattenGen().encode(chromosome)
            elif last_gen_type in layers2D:
                chromosome = FlattenGen().encode(chromosome)

        if len(chromosome) > 0 and not chromosome[-1] == GenType.Activation:
            chromosome = encoding_map[GenType.Activation].encode(chromosome)

        chromosome = encoding_map[GenType.OutputDense].encode(chromosome)

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
