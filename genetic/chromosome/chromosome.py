from genetic.gens.gen import Gen
from genetic.gen_type import GenType
from genetic.gens.gen_dense import DenseGen
from genetic.gens.gen_convolution_2d import Convolution2DGen


class Chromosome(object):
    """
    Represents chromosome, which is a set of gens.
    :param initial_gens - initial set of gens for current chromosome.
    """

    def __init__(self, initial_gens=None, constraints=None):

        if initial_gens is None:
            initial_gens = []

        if constraints is None:
            constraints = []

        self.gens = initial_gens
        self.constraints = constraints

    def __len__(self):
        return len(self.gens)

    def attach(self, gen):
        """
        Adds gen to set of gens for current chromosome.
        :param gen: gen to be added.
        :return: new instance of Chromosome object which contains additional gen.
        """
        if not isinstance(gen, Gen):
            raise Exception("Chromosome should contain only gen which is inherited from Gen object")

        for constraint in filter(lambda c: gen.type in c.target_type, self.constraints):
            constraint.evaluate(self.gens)

        self.gens.append(gen)

        return Chromosome(initial_gens=self.gens, constraints=self.constraints)

    def cross(self, chromosome):
        """
        Performs crossover operator for two parents.
        Operator produces two children with genotype of both parents.
        :param chromosome: the chromosome to be applied crossover operator.
        :return: two chromosomes which is a children of two parents.
        """
        crossover_point_for_first = self.index_of(GenType.Flatten)
        crossover_point_for_second = chromosome.index_of(GenType.Flatten)

        child_first = Chromosome(constraints=self.constraints)
        for gen in self.gens[:crossover_point_for_first]:
            child_first.attach(gen)
        for gen in chromosome.gens[crossover_point_for_second:]:
            child_first.attach(gen)

        child_second = Chromosome(constraints=self.constraints)
        for gen in chromosome.gens[:crossover_point_for_second]:
            child_second.attach(gen)
        for gen in self.gens[crossover_point_for_first:]:
            child_second.attach(gen)

        return child_first, child_second

    def mutate(self):
        """
        Applies mutation operator for current chromosome object.
        :return: mutated chromosome.
        """
        if len(self.gens) > 0:

            target_gens = list(self.gens)

            flatten_gen_index = self.index_of(GenType.Flatten)

            left_part_gens_count = len(target_gens[:flatten_gen_index])
            right_part_gens_count = len(target_gens[(flatten_gen_index + 1):])

            if left_part_gens_count < right_part_gens_count:
                is_2d = True
                start_index = 1
                end_index = flatten_gen_index
            else:
                is_2d = False
                start_index = flatten_gen_index + 1
                end_index = len(target_gens) - 1

            if (end_index - start_index) < 3:
                if is_2d:
                    target_gens.insert(end_index, Convolution2DGen())
                else:
                    target_gens.insert(end_index, DenseGen())

            elif (end_index - start_index) > 5:
                target_gens.pop(end_index)

        return Chromosome(initial_gens=target_gens, constraints=self.constraints)

    def contains_type(self, gen_type):
        return gen_type in map(lambda gen: gen.type, self.gens)

    def is_type_of(self, index, gen_type):
        return self.gens[index].type == gen_type

    def index_of(self, gen_type):
        return map(lambda gen: gen.type, self.gens).index(gen_type)
