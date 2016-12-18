import datetime
import numpy as np

from keras.datasets import cifar10

from genetic.estimation.accurancy_estimation import AccuracyEstimation
from genetic.gens import *
from genetic.model_genetic_excel import XlsxGeneticClassificationModel
from genetic.strategies import *
from genetic.chromo.chromosome import Chromosome

from pyeasyga.pyeasyga import GeneticAlgorithm


class GeneticClassificationModel(object):
    """
    Performs evolution model for classification.
    First step is creation a some set of initial neural networks.
    Gen structure is array of neurons for each or layers in the target neural network.
    """

    def __init__(self, estimation, population_size, generations):

        self.estimation = estimation
        self.fitness_callbacks = []

        self.attach_strategies = [
            DenseAttachStrategy(),
            ActivationAttachStrategy(),
            Convolution2dAttachStrategy(),
            OutputDenseAttachStrategy(),
            InputConvolution2dAttachStrategy()
        ]

        self.encoding_map = {
            GenType.Dense: lambda: DenseGen(),
            GenType.Flatten: lambda: FlattenGen(),
            GenType.Activation: lambda: ActivationGen(),
            GenType.Convolution2d: lambda: Convolution2DGen(),
            GenType.OutputDense: lambda: OutputDenseGen(size=10),
            GenType.OutputActivation: lambda: OutputActivation(),
            GenType.InputConvolution2DGen: lambda: InputConvolution2DGen(shape=(3, 32, 32)),
        }

        self.genetic = GeneticAlgorithm(
            seed_data=[],
            population_size=population_size,
            generations=generations,
            mutation_probability=0.5)

        self.genetic.mutate_function = self._mutation
        self.genetic.fitness_function = self._fitness
        self.genetic.crossover_function = self._crossover
        self.genetic.create_individual = self._create_individual

    def fit(self):
        """
        Fits appropriate model to find best individual
        among sets of neural network models.
        :return: best neural network model.
        """
        self.genetic.run()
        return self.genetic.best_individual()

    def add_callback(self, callback):
        self.fitness_callbacks.append(callback)

    def _fitness(self, member, _):
        """
        Calculates score for each of members of current population.
        :param member: the member of current population.
        :return: score value.
        """
        print(member)

        ratio = self.estimation.estimate(member)

        for callback in self.fitness_callbacks:
            callback(member, ratio)

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
        gen_seq += [random.choice(auto_generated_types) for _ in range(0, random.randint(2, 6))]
        gen_seq += [GenType.OutputDense, GenType.OutputActivation]

        for layer_type in gen_seq:
            for strategy in self.attach_strategies:
                if strategy.target_type == layer_type:
                    strategy.evaluate(chromosome, self.encoding_map[layer_type]())
                    break

        return chromosome


if __name__ == "__main__":

    def _to_expected_output(expected_output):
        class_number = expected_output[0]
        result = np.zeros(10, dtype=np.int)
        result[class_number] = 1
        return result

    training_set, _ = cifar10.load_data()
    inputs, expected_outputs = training_set

    np_input = np.array(inputs[:3])
    np_expected = np.array(map(_to_expected_output, expected_outputs)[:3])

    estimation = AccuracyEstimation(np_input, np_expected)
    model = GeneticClassificationModel(estimation, population_size=3, generations=1)

    xlsx_model = XlsxGeneticClassificationModel(
        model,
        "model-{0}.xlsx".format(datetime.datetime.now()).replace(":", "-"))
    xlsx_model.fit()

    xlsx_model.persist()
