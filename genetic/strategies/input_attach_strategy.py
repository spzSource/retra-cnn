from genetic.gen_type import GenType
from genetic.gens.gen_activation import ActivationGen
from genetic.strategies.attach_strategy import AttachStrategy


class InputConvolution2dAttachStrategy(AttachStrategy):
    @property
    def target_type(self):
        return GenType.InputConvolution2DGen

    def evaluate(self, chromosome, gen):
        chromosome = chromosome \
            .attach(gen) \
            .attach(ActivationGen())
        return chromosome
