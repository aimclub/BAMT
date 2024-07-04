from typing import Tuple

from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.operator import PopulationT
from golem.core.optimisers.genetic.parameters.mutation_prob import AdaptiveMutationProb
from golem.core.optimisers.genetic.parameters.parameter import AdaptiveParameter, ConstParameter

VariationOperatorProb = AdaptiveParameter[Tuple[float, float]]


class AdaptiveVariationProb(VariationOperatorProb):
    """Adaptive parameter for variation operators.
    Specifies mutation and crossover probabilities."""

    def __init__(self, mutation_prob: float = 0.5, crossover_prob: float = 0.5):
        self._mutation_prob_param = AdaptiveMutationProb(mutation_prob)
        self._mutation_prob = self._mutation_prob_param.initial
        self._crossover_prob_init = crossover_prob
        self._crossover_prob = crossover_prob

    @property
    def mutation_prob(self):
        return self._mutation_prob

    @property
    def crossover_prob(self):
        return self._crossover_prob

    @property
    def initial(self) -> Tuple[float, float]:
        return self._mutation_prob_param.initial, self._crossover_prob_init

    def next(self, population: PopulationT) -> Tuple[float, float]:
        self._mutation_prob = self._mutation_prob_param.next(population)
        self._crossover_prob = 1. - self._mutation_prob
        return self._mutation_prob, self._crossover_prob


def init_adaptive_operators_prob(parameters: GPAlgorithmParameters) -> VariationOperatorProb:
    """Returns static or adaptive parameter for mutation & crossover probabilities depending on genetic type scheme."""
    if parameters.genetic_scheme_type == GeneticSchemeTypesEnum.parameter_free:
        operators_prob = AdaptiveVariationProb()
    else:
        operators_prob = ConstParameter((parameters.mutation_prob, parameters.crossover_prob))
    return operators_prob
