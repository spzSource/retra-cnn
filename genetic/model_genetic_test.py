import unittest
import numpy as np

from keras.datasets import cifar10

from genetic.model_genetic import GeneticClassificationModel
from genetic.estimation.accurancy_estimation import AccuracyEstimation


def _to_expected_output(expected_output):
    class_number = expected_output[0]
    result = np.zeros(10, dtype=np.int)
    result[class_number] = 1
    return result


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
