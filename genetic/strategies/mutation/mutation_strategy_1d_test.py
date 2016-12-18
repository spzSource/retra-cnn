import unittest

from genetic.gens import *

from genetic.chromo.chromosome import Chromosome
from genetic.strategies.mutation import MutationStrategy1d


class MutationStrategy1dTest(unittest.TestCase):
    def test_mutation_by_insertion(self):
        chromosome = Chromosome([
            Convolution2DGen(),
            ActivationGen(),
            FlattenGen(),
            DenseGen()
        ])

        mutated = MutationStrategy1d(threshold=2).evaluate(chromosome)

        self.assertListEqual(
            map(lambda g: g.type, mutated),
            [
                GenType.Convolution2d,
                GenType.Activation,
                GenType.Flatten,
                GenType.Dense,
                GenType.Activation,
                GenType.Dense
            ])

    def test_check_to_be_mutated_case(self):
        chromosome = Chromosome([
            Convolution2DGen(),
            ActivationGen(),
            FlattenGen(),
            DenseGen()
        ])

        to_be_mutated = MutationStrategy1d().check(chromosome)

        self.assertTrue(to_be_mutated)

    def test_check_negative(self):
        chromosome = Chromosome([
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

        to_be_mutated = MutationStrategy1d().check(chromosome)

        self.assertFalse(to_be_mutated)
