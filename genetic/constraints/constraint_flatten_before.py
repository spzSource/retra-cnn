from genetic.gen_type import GenType
from genetic.constraints.constraint import Constraint
from genetic.gens.gen_flatten import FlattenGen


class FlattenBeforeConstraint(Constraint):

    @property
    def target_types(self):
        return [GenType.Dense]

    def __init__(self):
        self.encoding2d_map = [
            GenType.AvgPooling2d,
            GenType.Convolution2d,
            GenType.InputConvolution2DGen
        ]

    def evaluate(self, chromosome):
        chromosome_copy = list(chromosome)

        if len(chromosome_copy) > 0:
            target_gen = self._find_last_gen_except_activation(chromosome_copy)
            chromosome_copy = self._evaluate_for_target_gen(target_gen, chromosome_copy)

        return chromosome_copy

    def _evaluate_for_target_gen(self, gen, chromosome):
        if gen.type in self.encoding2d_map:
            chromosome.append(FlattenGen())
        return chromosome

    @staticmethod
    def _find_last_gen_except_activation(chromosome):
        target_gen = chromosome[-1]

        if target_gen.type == GenType.Activation and len(chromosome) > 1:
            target_gen = chromosome[-2]

        return target_gen
