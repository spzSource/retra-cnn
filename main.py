from classifier import *

if __name__ == "__main__":

    model_output = ".\models"

    model = Cifar10ClassificationModel(
        origin=PersistableClassificationModel(
            output_dir=model_output,
            origin=ClassificationModel()))

    model.build().train().persist()
