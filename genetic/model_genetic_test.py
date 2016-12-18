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

        np_input = np.array(inputs[:3])
        np_expected = np.array(map(_to_expected_output, expected_outputs)[:3])

        estimation = AccuracyEstimation(np_input, np_expected)
        self.model = GeneticClassificationModel(estimation, population_size=3, generations=1)

    # def test_run(self):
    #     self.model.fit()

    # def test_persist(self):
    #     xlsx_model = XlsxGeneticClassificationModel(
    #         self.model,
    #         "model-{0}.xlsx".format(datetime.datetime.now()).replace(":", "-"))
    #
    #     xlsx_model.fit()
    #     xlsx_model.persist()

if __name__ == '__main__':
    unittest.main()
