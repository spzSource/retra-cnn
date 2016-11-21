import unittest

from genetic.model_genetic import GeneticClassificationModel


class GeneticClassificationModelTest(unittest.TestCase):

    def setUp(self):
        self.model = GeneticClassificationModel()

    def test_create_individual(self):
        self.model.fit()


if __name__ == '__main__':
    unittest.main()
