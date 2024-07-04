from typing import TYPE_CHECKING

from golem.core.optimisers.genetic.operators.operator import PopulationT, Operator
from golem.core.optimisers.genetic.operators.selection import Selection
from golem.utilities.data_structures import ComparableEnum as Enum

if TYPE_CHECKING:
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters


class GeneticSchemeTypesEnum(Enum):
    steady_state = 'steady_state'
    generational = 'generational'
    parameter_free = 'parameter_free'


class Inheritance(Operator):
    def __init__(self, parameters: 'GPAlgorithmParameters', selection: Selection):
        super().__init__(parameters=parameters)
        self.selection = selection

    def __call__(self, previous_population: PopulationT, new_population: PopulationT) -> PopulationT:
        gp_scheme = self.parameters.genetic_scheme_type
        if gp_scheme == GeneticSchemeTypesEnum.generational:
            # Previous population is completely substituted
            next_population = self.direct_inheritance(new_population)
        elif gp_scheme == GeneticSchemeTypesEnum.steady_state:
            # Previous population is mixed with new one
            next_population = self.steady_state_inheritance(previous_population, new_population)
        elif gp_scheme == GeneticSchemeTypesEnum.parameter_free:
            # Same as steady-state
            next_population = self.steady_state_inheritance(previous_population, new_population)
        else:
            raise ValueError(f'Unknown genetic scheme {gp_scheme}!')
        return next_population

    def steady_state_inheritance(self,
                                 prev_population: PopulationT,
                                 new_population: PopulationT
                                 ) -> PopulationT:
        # use individuals with non-repetitive uid
        not_repetitive_inds = [ind for ind in prev_population if ind not in new_population]
        full_population = list(new_population) + list(not_repetitive_inds)
        return self.selection(full_population,
                              pop_size=self.parameters.pop_size)

    def direct_inheritance(self, new_population: PopulationT) -> PopulationT:
        return new_population[:self.parameters.pop_size]
