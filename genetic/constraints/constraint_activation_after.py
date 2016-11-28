from genetic.constraints.constraint import Constraint
from genetic.gen_type import GenType
from genetic.gens.gen_activation import ActivationGen


class ActivationAfterConstraint(Constraint):
    @property
    def target_types(self):
        return [
            GenType.Convolution2d,
            GenType.InputConvolution2DGen,
            GenType.Dense,
            GenType.AvgPooling2d
        ]

    def evaluate(self, chromosome):
        if len(chromosome) > 0:
            last_gen = chromosome[-1]
            if not last_gen.type == GenType.Activation:
                chromosome.append(ActivationGen())
        return chromosome
