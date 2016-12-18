from classifier import Classification


class VisualizedModel(Classification):
    
    def __init__(self, origin):
        self.origin = origin

    @property
    def model(self):
        return self.origin.model

    def build(self):
        self.origin.build()
        return self

    def train(self, training_set):
        self.origin.train(training_set)
        return self

    def show(self, file_path):
        print(self.origin.model)
        from keras.utils.visualize_util import plot
        plot(self.origin.model, show_shapes=True, to_file=file_path)

        from PIL import Image
        img = Image.open(file_path)
        img.show()
