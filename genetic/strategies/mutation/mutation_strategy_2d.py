from genetic.chromo.chromosome import Chromosome
from genetic.gen_type import GenType
from genetic.gens import ActivationGen
from genetic.gens.gen_convolution_2d import Convolution2DGen
from genetic.strategies.mutation.mutation_strategy import MutationStrategy


class MutationStrategy2d(MutationStrategy):
    """
    Strategy which performs mutation for two-dimension part of layers.
    """
    def __init__(self, threshold=3):
        """
        :type threshold: the threshold which determines
        whether we should delete gen or insert a new one.
        """
        self.threshold = threshold

    def check(self, chromosome):
        left_count, right_count = self._calculate_parts_size(chromosome)
        return left_count < right_count

    def evaluate(self, chromosome):
        target_gens = list(chromosome.gens)

        start_index = 1
        end_index = lambda: chromosome.index_of(GenType.Flatten)

        if (end_index() - start_index) <= 3:
            if target_gens[end_index()].type == GenType.Activation:
                target_gens.insert(end_index(), Convolution2DGen())
                target_gens.insert(end_index(), ActivationGen())
            else:
                target_gens.insert(end_index(), ActivationGen())
                target_gens.insert(end_index(), Convolution2DGen())
        elif (end_index() - start_index) <= 5:
            target_gens.pop(end_index())

        return Chromosome(target_gens)

    @staticmethod
    def _calculate_parts_size(chromosome):
        flatten_gen_index = chromosome.index_of(GenType.Flatten)
        left_count = len(chromosome.gens[:flatten_gen_index])
        right_count = len(chromosome.gens[(flatten_gen_index + 1):])

        return left_count, right_count
