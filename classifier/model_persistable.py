import os
import datetime


class PersistableClassificationModel(object):

    """
    Classification classifier with ability to persist trained classifier on the disk.
    """
    def __init__(self, output_dir, origin):
        self.originModel = origin
        self.path_to_persist = os.path.join(output_dir, '{0}/classifier-{0}.mdl'.format(datetime.datetime.now()))

    def persist(self):
        """
        Persists original classifier to the file.
        """
        self.originModel.model.save(self.path_to_persist)
        return self

    def build(self):
        """
        Simply calls original classifier to build classifier.
        """
        self.originModel.build()
        return self

    def train(self, training_set):
        """
        Simply calls original classifier to train classifier.
        """
        self.originModel.train(training_set)
        return self
