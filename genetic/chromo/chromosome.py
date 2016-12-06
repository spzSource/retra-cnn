from functools import reduce

from genetic.gen_type import GenType
from genetic.gens.gen import Gen


class Chromosome(object):
    """
    Represents chromo, which is a set of gens.
    :param initial_gens - initial set of gens for current chromo.
    """

    def __init__(self, initial_gens=None, mutations=None):

        if initial_gens is None:
            initial_gens = []

        if mutations is None:
            import genetic.strategies.mutation

            mutations = [
                genetic.strategies.mutation.MutationStrategy1d(),
                genetic.strategies.mutation.MutationStrategy2d()
            ]

        self.gens = initial_gens
        self.mutations = mutations

    def attach(self, gen):
        """
        Adds gen to set of gens for current chromo.
        :param gen: gen to be added.
        :return: new instance of Chromosome object which contains additional gen.
        """
        if not isinstance(gen, Gen):
            raise Exception("Chromosome should contain only gen which is inherited from Gen object")

        self.gens.append(gen)

        return Chromosome(initial_gens=self.gens)

    def cross(self, chromosome):
        """
        Performs crossover operator for two parents.
        Operator produces two children with genotype of both parents.
        :param chromosome: the chromo to be applied crossover operator.
        :return: two chromosomes which is a children of two parents.
        """
        crossover_point_for_first = self.index_of(GenType.Flatten)
        crossover_point_for_second = chromosome.index_of(GenType.Flatten)

        child_first = reduce(
            lambda res, item: res.attach(item),
            self.gens[:crossover_point_for_first],
            Chromosome())

        child_first = reduce(
            lambda res, item: res.attach(item),
            chromosome.gens[crossover_point_for_second:],
            child_first)

        child_second = reduce(
            lambda res, item: res.attach(item),
            chromosome.gens[:crossover_point_for_second],
            Chromosome())

        child_second = reduce(
            lambda res, item: res.attach(item),
            self.gens[crossover_point_for_first:],
            child_second)

        return child_first, child_second

    def mutate(self):
        """
        Applies mutation operator for current chromo object.
        :return: mutated chromo.
        """
        target_gens = []

        if len(self.gens) > 0:
            for mutation in self.mutations:
                if mutation.check(self):
                    mutation.evaluate(self)
                    break

        return Chromosome(initial_gens=target_gens)

    def contains_type(self, gen_type):
        """
        Indicates whether current chromo contains gen with specified type.
        :param gen_type: The type of required gen.
        :return: True - contains, otherwise - False
        """
        return gen_type in map(lambda gen: gen.type, self.gens)

    def is_type_of(self, index, gen_type):
        """
        Check the type of gen by index.
        :param index: the index of gen in the chromo.
        :param gen_type: The required get to check.
        :return: True - type is matched, otherwise - False.
        """
        return self.gens[index].type == gen_type

    def index_of(self, gen_type):
        """
        Returns an index of gen with specified type inside chromo.
        :param gen_type: Required type of gen.
        :return: index of gen.
        """
        return list(map(lambda gen: gen.type, self.gens)).index(gen_type)

    def __len__(self):
        """
        Returns length of chromo.
        """
        return len(self.gens)

    def __str__(self):
        """
        Returns string representation of current chromo.
        """
        return "---\n" + "\n".join(map(lambda gen: str(gen), self.gens)) + "\n---"
