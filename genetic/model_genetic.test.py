import unittest

from genetic.model_genetic import GeneticClassificationModel


class GeneticClassificationModelTest(unittest.TestCase):
    def setUp(self):
        self.model = GeneticClassificationModel()

    # def test_model(self):
    #     self.model.fit()

    def test_crossover_for_parents_with_flatten_gen(self):
        """
        Compares two produced children after crossover operator applying.
        Parent chromosomes both have flatten gen inside.
        :return:
        """
        first_parent = [
            (6, [8, 16]),
            (2, []),
            (1, [8, 8]),
            (2, []),
            (1, [16, 4]),
            (2, []),
            (5, []),
            (3, [1024]),
            (7, [10])
        ]

        second_parent = [
            (6, [4, 8]),
            (2, []),
            (1, [32, 4]),
            (2, []),
            (5, []),
            (3, [2048]),
            (7, [10])
        ]

        (child_first, child_second) = self.model._crossover(first_parent, second_parent)

        self.assertIsNotNone(child_first)
        self.assertIsNotNone(second_parent)

        self.assertListEqual([
            (6, [8, 16]),
            (2, []),
            (1, [8, 8]),
            (2, []),
            (1, [16, 4]),
            (2, []),
            (5, []),
            (3, [2048]),
            (7, [10])
        ], child_first)

        self.assertListEqual([
            (6, [4, 8]),
            (2, []),
            (1, [32, 4]),
            (2, []),
            (5, []),
            (3, [1024]),
            (7, [10])
        ], child_second)


if __name__ == '__main__':
    unittest.main()
