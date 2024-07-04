from typing import Optional

import numpy as np

from golem.core.constants import MIN_POP_SIZE, EVALUATION_ATTEMPTS_NUMBER
from golem.core.log import default_log
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import Crossover
from golem.core.optimisers.genetic.operators.mutation import Mutation
from golem.core.optimisers.genetic.operators.operator import PopulationT, EvaluationOperator
from golem.core.optimisers.genetic.operators.selection import Selection
from golem.core.optimisers.populational_optimizer import EvaluationAttemptsError
from golem.utilities.data_structures import ensure_wrapped_in_sequence


class ReproductionController:
    """
    Task of the Reproduction Controller is to reproduce population
    while keeping population size as specified in optimizer settings.

    It implements a simple proportional controller that compensates for
    invalid results each generation by computing average ratio of valid results.
    Invalid results include cases when Operators, Evaluator or GraphVerifier
    return output population that's smaller than the input population.

    Example.
    Let's say we need a population of size 50. Let's say about 20% of individuals
    are *usually* evaluated with an error. If we take select only 50 for the new population,
    we will get about 40 valid ones. Not enough. Therefore, we need to take more.
    How much more? Approximately by `target_pop_size / mean_success_rate = 50 / 0.8 ~= 62'.
    Here `mean_success_rate` estimates number of successfully evaluated individuals.
    Then we request 62, then approximately 62*0.8~=50 of them are valid in the end,
    and we achieve target size more reliably. This runs in a loop to control stochasticity.

    Args:
        parameters: genetic algorithm parameters.
        selection: operator used in reproduction.
        mutation: operator used in reproduction.
        crossover: operator used in reproduction.
        window_size: size in iterations of the moving window to compute reproduction success rate.
    """

    def __init__(self,
                 parameters: GPAlgorithmParameters,
                 selection: Selection,
                 mutation: Mutation,
                 crossover: Crossover,
                 window_size: int = 10,
                 ):
        self.parameters = parameters
        self.selection = selection
        self.mutation = mutation
        self.crossover = crossover

        self._minimum_valid_ratio = parameters.required_valid_ratio * 0.5
        self._window_size = window_size
        self._success_rate_window = np.full(self._window_size, 1.0)

        self._log = default_log(self)

    @property
    def mean_success_rate(self) -> float:
        """Returns mean success rate of reproduction + evaluation,
        fraction of how many individuals were reproduced and mutated successfully.
        Computed as average fraction for the last N iterations (N = window size param)"""
        return float(np.mean(self._success_rate_window))

    def reproduce_uncontrolled(self,
                               population: PopulationT,
                               evaluator: EvaluationOperator,
                               pop_size: Optional[int] = None,
                               ) -> PopulationT:
        """Reproduces and evaluates population (select, crossover, mutate).
        Doesn't implement any additional checks on population.
        """
        # If operators can return unchanged individuals from previous population
        # (e.g. both Mutation & Crossover are not applied with some probability)
        # then there's a probability that duplicate individuals can appear

        # TODO: it can't choose more than len(population)!
        #  It can be faster if it could.
        selected_individuals = self.selection(population, pop_size)
        new_population = self.crossover(selected_individuals)
        new_population = ensure_wrapped_in_sequence(self.mutation(new_population))
        new_population = evaluator(new_population)
        return new_population

    def reproduce(self,
                  population: PopulationT,
                  evaluator: EvaluationOperator
                  ) -> PopulationT:
        """Reproduces and evaluates population (select, crossover, mutate).
        Implements additional checks on population to ensure that population size
        follows required population size.
        """
        total_target_size = self.parameters.pop_size  # next population size
        collected_next_population = {}
        for i in range(EVALUATION_ATTEMPTS_NUMBER):
            # Estimate how many individuals we need to complete new population
            # based on average success rate of valid results
            residual_size = total_target_size - len(collected_next_population)
            residual_size = max(MIN_POP_SIZE,
                                int(residual_size / self.mean_success_rate))
            residual_size = min(len(population), residual_size)

            # Reproduce the required number of individuals that equals residual size
            partial_next_population = self.reproduce_uncontrolled(population, evaluator, residual_size)
            # Avoid duplicate individuals that can come unchanged from previous population
            collected_next_population.update({ind.uid: ind for ind in partial_next_population})

            # Keep running average of transform success rate (if sample is big enough)
            if len(partial_next_population) >= MIN_POP_SIZE:
                valid_ratio = len(partial_next_population) / residual_size
                self._success_rate_window = np.roll(self._success_rate_window, shift=1)
                self._success_rate_window[0] = valid_ratio

            # Successful return: got enough individuals
            if len(collected_next_population) >= total_target_size * self.parameters.required_valid_ratio:
                self._log.info(f'Reproduction achieved pop size {len(collected_next_population)}'
                               f' using {i+1} attempt(s) with success rate {self.mean_success_rate:.3f}')
                return list(collected_next_population.values())[:total_target_size]
        else:
            # If number of evaluation attempts is exceeded return a warning or raise exception
            helpful_msg = ('Check objective, constraints and evo operators. '
                           'Possibly they return too few valid individuals.')

            if len(collected_next_population) >= total_target_size * self._minimum_valid_ratio:
                self._log.warning(f'Could not achieve required population size: '
                                  f'have {len(collected_next_population)},'
                                  f' required {total_target_size}!\n' + helpful_msg)
                return list(collected_next_population.values())
            else:
                raise EvaluationAttemptsError('Could not collect valid individuals'
                                              ' for next population.' + helpful_msg)
