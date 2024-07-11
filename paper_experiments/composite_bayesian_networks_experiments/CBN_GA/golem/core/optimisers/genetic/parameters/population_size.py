import math
from typing import Optional

from golem.core.constants import MIN_POP_SIZE
from golem.core.optimisers.archive.generation_keeper import ImprovementWatcher
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.operator import PopulationT
from golem.core.optimisers.genetic.parameters.parameter import AdaptiveParameter
from golem.core.utilities.data_structures import BidirectionalIterator
from golem.core.utilities.sequence_iterator import fibonacci_sequence, SequenceIterator

PopulationSize = AdaptiveParameter[int]


class ConstRatePopulationSize(PopulationSize):
    def __init__(self, pop_size: int, offspring_rate: float, max_pop_size: Optional[int] = None):
        self._offspring_rate = offspring_rate
        self._initial = pop_size
        self._max_pop_size = max_pop_size

    @property
    def initial(self) -> int:
        return self._initial

    def next(self, population: PopulationT) -> int:
        # to prevent stagnation
        pop_size = max(len(population), self._initial)
        if not self._max_pop_size or pop_size < self._max_pop_size:
            pop_size += math.ceil(pop_size * self._offspring_rate)
        if self._max_pop_size:
            pop_size = min(pop_size, self._max_pop_size)
        return pop_size


class AdaptivePopulationSize(PopulationSize):
    def __init__(self,
                 improvement_watcher: ImprovementWatcher,
                 progression_iterator: BidirectionalIterator[int],
                 max_pop_size: Optional[int] = None):
        self._improvements = improvement_watcher
        self._iterator = progression_iterator
        self._max_pop_size = max_pop_size
        self._initial = self._iterator.next() if self._iterator.has_next() else self._iterator.prev()

    @property
    def initial(self) -> int:
        return self._initial

    def next(self, population: PopulationT) -> int:
        pop_size = len(population)
        too_many_fitness_eval_errors = pop_size / self._iterator.current() < 0.5

        if too_many_fitness_eval_errors or not self._improvements.is_any_improved:
            if self._iterator.has_next():
                pop_size = self._iterator.next()
        elif self._improvements.is_quality_improved and self._improvements.is_complexity_improved and pop_size > 0:
            if self._iterator.has_prev():
                pop_size = self._iterator.prev()

        pop_size = max(pop_size, MIN_POP_SIZE)
        if self._max_pop_size:
            pop_size = min(pop_size, self._max_pop_size)

        return pop_size


def init_adaptive_pop_size(requirements: GPAlgorithmParameters,
                           improvement_watcher: ImprovementWatcher) -> PopulationSize:
    genetic_scheme_type = requirements.genetic_scheme_type
    if genetic_scheme_type == GeneticSchemeTypesEnum.steady_state:
        pop_size = ConstRatePopulationSize(
            pop_size=requirements.pop_size,
            offspring_rate=1.0,
            max_pop_size=requirements.max_pop_size,
        )
    elif genetic_scheme_type == GeneticSchemeTypesEnum.generational:
        pop_size = ConstRatePopulationSize(
            pop_size=requirements.pop_size,
            offspring_rate=requirements.offspring_rate,
            max_pop_size=requirements.max_pop_size,
        )
    elif genetic_scheme_type == GeneticSchemeTypesEnum.parameter_free:
        pop_size_progression = SequenceIterator(sequence_func=fibonacci_sequence,
                                                start_value=requirements.pop_size,
                                                min_sequence_value=1,
                                                max_sequence_value=requirements.max_pop_size)
        pop_size = AdaptivePopulationSize(improvement_watcher=improvement_watcher,
                                          progression_iterator=pop_size_progression,
                                          max_pop_size=requirements.max_pop_size)
    else:
        raise ValueError(f"Unknown genetic type scheme {genetic_scheme_type}")
    return pop_size
