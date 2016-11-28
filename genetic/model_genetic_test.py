import unittest

from genetic.chromosome import Chromosome
from genetic.gen_type import GenType
from genetic.gens.gen_activation import ActivationGen
from genetic.gens.gen_convolution_2d import Convolution2DGen
from genetic.gens.gen_dense import DenseGen
from genetic.gens.gen_flatten import FlattenGen
from genetic.model_genetic import GeneticClassificationModel

#
# class ChromosomeTest(unittest.TestCase):
#
#     def setUp(self):
#         self.chromosome = Chromosome()
#
#     def test_immutability(self):
#         new_chromosome = self.chromosome.push_back(Convolution2DGen())
#         self.assertNotEqual(new_chromosome, self.chromosome)
#
#     def test_chromosome_length(self):
#         new_chromosome = self.chromosome\
#             .push_back(Convolution2DGen())\
#             .push_back(ActivationGen())\
#             .push_back(Convolution2DGen())
#
#         self.assertEqual(3, len(new_chromosome))
#
#     def test_mutation_operator(self):
#         origin = Chromosome(initial_gens=[
#             Convolution2DGen(),
#             ActivationGen(),
#             Convolution2DGen(),
#             ActivationGen(),
#             FlattenGen(),
#             DenseGen(),
#             ActivationGen()
#         ])
#
#         mutated = origin.mutate()
#
#         self.assertNotEqual(origin, mutated)
#
#     def test_cross_operator(self):
#         first = Chromosome(initial_gens=[
#             Convolution2DGen(),
#             ActivationGen(),
#             Convolution2DGen(),
#             ActivationGen(),
#             FlattenGen(),
#             DenseGen(),
#             ActivationGen()
#         ])
#
#         second = Chromosome(initial_gens=[
#             Convolution2DGen(),
#             ActivationGen(),
#             FlattenGen(),
#             DenseGen(),
#             ActivationGen(),
#             DenseGen(),
#             ActivationGen(),
#             DenseGen(),
#             ActivationGen()
#         ])
#
#         (first_child, second_child) = first.cross(second)
#
#         self.assertEqual(11, len(first_child))
#         self.assertEqual(5, len(second_child))


class GeneticClassificationModelTest(unittest.TestCase):
    def setUp(self):
        self.model = GeneticClassificationModel()

    def test_run(self):
        self.model.fit()

if __name__ == '__main__':
    unittest.main()
