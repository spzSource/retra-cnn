from genetic.gen_type import GenType
from genetic.gens.gen_activation import ActivationGen
from genetic.strategies.attach_strategy import AttachStrategy


class Convolution2dAttachStrategy(AttachStrategy):
    @property
    def target_type(self):
        return GenType.Convolution2d

    def evaluate(self, chromosome, gen):
        if not chromosome.contains_type(GenType.Dense):
            chromosome = chromosome\
                .attach(gen)\
                .attach(ActivationGen())
        return chromosome
