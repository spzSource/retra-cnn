from genetic.gen_type import GenType
from genetic.gens.gen_flatten import FlattenGen
from genetic.gens.gen_activation import ActivationGen
from genetic.strategies.attach_strategy import AttachStrategy


class DenseAttachStrategy(AttachStrategy):
    @property
    def target_type(self):
        return GenType.Dense

    def evaluate(self, chromosome, gen):
        if not chromosome.contains_type(self.target_type):
            chromosome = chromosome.attach(FlattenGen())
        chromosome = chromosome\
            .attach(gen)\
            .attach(ActivationGen())
        return chromosome
