from genetic.chromosome import Chromosome
from genetic.gen_type import GenType
from genetic.gens.gen_convolution_2d import Convolution2DGen
from genetic.strategies.mutation.mutation_strategy import MutationStrategy


class MutationStrategy2d(MutationStrategy):

    def check(self, chromosome):
        left_count, right_count = self._calculate_parts_size(chromosome)
        return left_count < right_count

    def evaluate(self, chromosome):
        target_gens = list(chromosome.gens)

        start_index = 1
        end_index = chromosome.index_of(GenType.Flatten)

        if (end_index - start_index) <= 3:
            target_gens.insert(end_index, Convolution2DGen())
        else:
            target_gens.pop(end_index)

        return Chromosome(target_gens)

    @staticmethod
    def _calculate_parts_size(chromosome):
        flatten_gen_index = chromosome.index_of(GenType.Flatten)
        left_count = len(chromosome.gens[:flatten_gen_index])
        right_count = len(chromosome.gens[(flatten_gen_index + 1):])

        return left_count, right_count
