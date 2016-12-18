from keras.models import Sequential
from keras.optimizers import SGD

from genetic.estimation.estimation import ChromosomeEstimation


class AccuracyEstimation(ChromosomeEstimation):

    def __init__(self, target_input, expected_output):
        self.target_input = target_input
        self.expected_output = expected_output

    def estimate(self, chromosome):
        """
        Estimates passed chromosome using Sequential model.
        :param chromosome: target chromosome to be estimated.
        :return: estimation value fot passed chromosome.
        """

        print("Current chromosome: {0}".format(str(chromosome)))

        layers = self._decode_chromosome(chromosome)

        try:
            model = Sequential(layers)
            model.compile(
                loss="mse",
                metrics=["accuracy"],
                optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False))

            history = model.fit(self.target_input, self.expected_output, batch_size=10, nb_epoch=100)
            ratio = history.history["acc"][-1]

        except ValueError as e:
            print(e)
            ratio = 0

        return ratio

    def _decode_chromosome(self, chromosome):
        """
        Converts encoded chromosome to array of layers.
        :param chromosome: array of encoded gens.
        :return: array of layers for neural network.
        """
        layers = []
        for current_gen in chromosome.gens:
            layer = current_gen.decode()
            layers.append(layer)
        return layers
