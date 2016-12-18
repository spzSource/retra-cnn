from keras.datasets import cifar10

from classifier import Classification


class Cifar10ClassificationModel(Classification):

    """
    Classification classifier with ability to persist trained classifier on the disk.
    """
    def __init__(self, origin):
        self.originModel = origin

    @property
    def model(self):
        return self.originModel.model

    def build(self):
        """
        Simply calls original classifier to build classifier.
        """
        self.originModel.build()
        return self

    def train(self, training_set=None):
        """
        Train original classifier using CIFAR-10 image data set.
        """
        if training_set is None:
            training_set, _ = cifar10.load_data()
        self.originModel.train(training_set)
        return self

    def persist(self):
        self.originModel.persist()
