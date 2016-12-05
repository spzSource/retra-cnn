import unittest

from genetic.gens import *
from genetic.gen_type import GenType
from genetic.chromo.chromosome import Chromosome


class ChromosomeTest(unittest.TestCase):

    def setUp(self):
        self.chromosome = Chromosome()

    def test_immutability(self):
        new_chromosome = self.chromosome.attach(Convolution2DGen())
        self.assertNotEqual(new_chromosome, self.chromosome)

    def test_chromosome_length(self):
        new_chromosome = self.chromosome\
            .attach(Convolution2DGen())\
            .attach(ActivationGen())\
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

        first_child, second_child = first.cross(second)

        self.assertEqual(11, len(first_child))
        self.assertEqual(5, len(second_child))

        self.assertTrue(
            first_child.contains_type(GenType.Flatten),
            "Chromosomes should contain Flatten gen.")

        self.assertListEqual(
            map(lambda gen: gen.type, first_child),
            [
                GenType.Convolution2d,
                GenType.Activation,
                GenType.Convolution2d,
                GenType.Activation,
                GenType.Flatten,
                GenType.Dense,
                GenType.Activation,
                GenType.Dense,
                GenType.Activation,
                GenType.Dense,
                GenType.Activation
            ])

        self.assertListEqual(
            map(lambda gen: gen.type, second_child),
            [
                GenType.Convolution2d,
                GenType.Activation,
                GenType.Flatten,
                GenType.Dense,
                GenType.Activation
            ])

if __name__ == '__main__':
    unittest.main()