from genetic.gen_type import GenType
from genetic.gens.gen_activation import OutputActivation
from genetic.gens.gen_flatten import FlattenGen
from genetic.gens.gen_output_dense import OutputDenseGen
from genetic.strategies.attach_strategy import AttachStrategy


class OutputDenseAttachStrategy(AttachStrategy):
    @property
    def target_type(self):
        return GenType.OutputDense

    def evaluate(self, chromosome, gen):
        if not chromosome.contains_type(GenType.Dense):
            chromosome = chromosome.attach(FlattenGen())
        chromosome = chromosome.attach(OutputDenseGen())
        chromosome = chromosome.attach(OutputActivation())
        return chromosome
