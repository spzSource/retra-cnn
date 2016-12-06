import unittest
import datetime
import numpy as np

from keras.datasets import cifar10

from genetic.chromo.chromosome import Chromosome
from genetic.estimation.accurancy_estimation import AccuracyEstimation
from genetic.gens.gen_activation import ActivationGen
from genetic.gens.gen_convolution_2d import Convolution2DGen
from genetic.gens.gen_dense import DenseGen
from genetic.gens.gen_flatten import FlattenGen
from genetic.model_genetic import GeneticClassificationModel
from genetic.model_genetic_excel import XlsxGeneticClassificationModel


def _to_expected_output(expected_output):
    class_number = expected_output[0]
    result = np.zeros(10, dtype=np.int)
    result[class_number] = 1
    return result


class ChromosomeTest(unittest.TestCase):
    def setUp(self):
        self.chromosome = Chromosome()

    def test_immutability(self):
        new_chromosome = self.chromosome.attach(Convolution2DGen())
        self.assertNotEqual(new_chromosome, self.chromosome)

    def test_chromosome_length(self):
        new_chromosome = self.chromosome \
            .attach(Convolution2DGen()) \
            .attach(ActivationGen()) \
            .attach(Convolution2DGen())

        self.assertEqual(3, len(new_chromosome))

    def test_mutation_operator(self):
        origin = Chromosome(initial_gens=[
            Convolution2DGen(),
            ActivationGen(),
            Convolution2DGen(),
            ActivationGen(),
            FlattenGen(),
            DenseGen(),
            ActivationGen()
        ])

        mutated = origin.mutate()

        self.assertNotEqual(origin, mutated)

    def test_cross_operator(self):
        first = Chromosome(initial_gens=[
            Convolution2DGen(),
            ActivationGen(),
            Convolution2DGen(),
            ActivationGen(),
            FlattenGen(),
            DenseGen(),
            ActivationGen()
        ])

        second = Chromosome(initial_gens=[
            Convolution2DGen(),
            ActivationGen(),
            FlattenGen(),
            DenseGen(),
            ActivationGen(),
            DenseGen(),
            ActivationGen(),
            DenseGen(),
            ActivationGen()
        ])

        (first_child, second_child) = first.cross(second)

        self.assertEqual(11, len(first_child))
        self.assertEqual(5, len(second_child))


class GeneticClassificationModelTest(unittest.TestCase):
    def setUp(self):
        training_set, _ = cifar10.load_data()
        inputs, expected_outputs = training_set

        np_input = np.array(inputs[:30])
        np_expected = np.array(list(map(_to_expected_output, expected_outputs))[:30])
        print(np_expected)

        estimation = AccuracyEstimation(np_input, np_expected)
        self.model = GeneticClassificationModel(estimation, population_size=10, generations=30)

    # def test_run(self):
    #     self.model.fit()

    def test_persist(self):
        xlsx_model = XlsxGeneticClassificationModel(
            self.model,
            "model-{0}.xlsx".format(datetime.datetime.now()).replace(":", "-"))

        fitness, gens = xlsx_model.fit()
        xlsx_model.persist()

        print("score = {0}, net = {1}".format(filter, str(gens)))


if __name__ == '__main__':
    unittest.main()
