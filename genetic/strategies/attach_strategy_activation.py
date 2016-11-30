from genetic.gen_type import GenType
from genetic.strategies.attach_strategy import AttachStrategy


class ActivationAttachStrategy(AttachStrategy):

    @property
    def target_type(self):
        return GenType.Activation

    def evaluate(self, chromosome, gen):
        if len(chromosome) > 0 and not chromosome.is_type_of(-1, GenType.Activation):
            chromosome = chromosome.attach(gen)
        return chromosome
