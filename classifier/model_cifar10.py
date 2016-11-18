from keras.datasets import cifar10


class Cifar10ClassificationModel(object):

    """
    Classification classifier with ability to persist trained classifier on the disk.
    """
    def __init__(self, origin):
        self.originModel = origin

    def build(self):
        """
        Simply calls original classifier to build classifier.
        """
        self.originModel.build()
        return self

    def train(self):
        """
        Train original classifier using CIFAR-10 image data set.
        """
        training_set, _ = cifar10.load_data()
        self.originModel.train(training_set)
        return self

    def persist(self):
        self.originModel.persist()
