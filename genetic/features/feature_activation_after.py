from genetic.features.feature import Feature
from genetic.gen_type import GenType
from genetic.gens.gen_activation import ActivationGen


class ActivationAfterFeature(Feature):
    def evaluate(self, chromosome):
        if len(chromosome) > 0:
            (gen_type, _) = chromosome[-1]
            if not gen_type == GenType.Activation:
                chromosome = ActivationGen().encode(list(chromosome))
        return chromosome
