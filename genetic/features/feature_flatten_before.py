from genetic.gen_type import GenType
from genetic.features.feature import Feature
from genetic.gens.gen_flatten import FlattenGen


class FlattenBeforeFeature(Feature):

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
        (gen_type, _) = gen
        if gen_type in self.encoding2d_map:
            chromosome = FlattenGen().encode(chromosome)
        return chromosome

    @staticmethod
    def _find_last_gen_except_activation(chromosome):
        target_gen = chromosome[-1]
        (gen_type, _) = target_gen

        if gen_type == GenType.Activation and len(chromosome) > 1:
            target_gen = chromosome[-2]

        return target_gen
