from dataclasses import dataclass
from typing import Sequence, Union, Any, Optional, Callable

from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.adaptive.mab_agents.neural_contextual_mab_agent import ContextAgentTypeEnum
from golem.core.optimisers.genetic.operators.base_mutations import MutationStrengthEnum, MutationTypesEnum, \
    simple_mutation_set
from golem.core.optimisers.optimizer import AlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.elitism import ElitismTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.regularization import RegularizationTypesEnum
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum


@dataclass
class GPAlgorithmParameters(AlgorithmParameters):
    """
    Defines parameters of evolutionary operators and the algorithm of genetic optimizer.

    :param crossover_prob: crossover probability (chance that two individuals will be mated).
    :param mutation_prob: mutation probability (chance that an individual will be mutated).
    :param variable_mutation_num: flag to apply mutation one or few times for individual in each iteration.
    :param max_num_of_operator_attempts: max number of unsuccessful evo operator attempts before continuing.
    :param mutation_strength: strength of mutation in tree (using in certain mutation types)
    :param min_pop_size_with_elitism: minimal population size with which elitism is applicable
    :param required_valid_ratio: ratio of valid individuals on next population to continue optimization.

    Used in `ReproductionController` to compensate for invalid individuals. See the class for details.

    :param adaptive_mutation_type: Experimental feature! Enables adaptive Mutation agent.
    :param context_agent_type: Experimental feature! Enables graph encoding for Mutation agent.

    Adaptive mutation agent uses specified algorithm. 'random' type is the default non-adaptive version.
    Requires crossover_types to be CrossoverTypesEnum.none for correct adaptive learning,
    so that fitness changes depend only on agent's actions (chosen mutations).
    ``MutationAgentTypeEnum.bandit`` uses Multi-Armed Bandit (MAB) learning algorithm.
    ``MutationAgentTypeEnum.contextual_bandit`` uses contextual MAB learning algorithm.
    ``MutationAgentTypeEnum.neural_bandit`` uses contextual MAB learning algorithm with Deep Neural encoding.

    Parameter `context_agent_type` specifies implementation of graph/node encoder for adaptive
    mutation agent. It is relevant for contextual and neural bandits.

    :param selection_types: Sequence of selection operators types
    :param crossover_types: Sequence of crossover operators types
    :param mutation_types: Sequence of mutation operators types
    :param elitism_type: type of elitism operator evolution

    :param regularization_type: type of regularization operator

    Regularization attempts to cut off the subtrees of the graph. If the truncated graph
    is not worse than the original, then it enters the new generation as a simpler solution.
    Regularization is not used by default, it must be explicitly enabled.

    :param genetic_scheme_type: type of genetic evolutionary scheme

    The `generational` scheme is a standard scheme of the evolutionary algorithm.
    It specifies that at each iteration the entire generation is updated.

    In the `steady_state` scheme at each iteration only one individual is updated.

    The `parameter_free` scheme is an adaptive variation of the `steady_state` scheme.
    It specifies that the population size and the probability of mutation and crossover
    change depending on the success of convergence. If there are no improvements in fitness,
    then the size and the probabilities increase. When fitness improves, the size and the
    probabilities decrease. That is, the algorithm choose a more stable and conservative
    mode when optimization seems to converge.

    :param decaying_factor: decaying factor for Multi-Armed Bandits for managing the profit from operators
        The smaller the value of decaying_factor, the larger the influence for the best operator.
    :param window_size: the size of sliding window for Multi-Armed Bandits to decrease variance.
        The window size is measured by the number of individuals to consider.
    """

    crossover_prob: float = 0.8
    mutation_prob: float = 0.8
    variable_mutation_num: bool = True
    max_num_of_operator_attempts: int = 100
    mutation_strength: MutationStrengthEnum = MutationStrengthEnum.mean
    min_pop_size_with_elitism: int = 5
    required_valid_ratio: float = 0.9

    adaptive_mutation_type: MutationAgentTypeEnum = MutationAgentTypeEnum.default
    context_agent_type: Union[ContextAgentTypeEnum, Callable] = ContextAgentTypeEnum.nodes_num

    selection_types: Optional[Sequence[Union[SelectionTypesEnum, Any]]] = None
    crossover_types: Sequence[Union[CrossoverTypesEnum, Any]] = \
        (CrossoverTypesEnum.one_point,)
    mutation_types: Sequence[Union[MutationTypesEnum, Any]] = simple_mutation_set
    elitism_type: ElitismTypesEnum = ElitismTypesEnum.keep_n_best
    regularization_type: RegularizationTypesEnum = RegularizationTypesEnum.none
    genetic_scheme_type: GeneticSchemeTypesEnum = GeneticSchemeTypesEnum.generational

    decaying_factor: float = 1.0
    window_size: Optional[int] = None

    def __post_init__(self):
        if not self.selection_types:
            self.selection_types = (SelectionTypesEnum.spea2,) if self.multi_objective \
                else (SelectionTypesEnum.tournament,)
        if self.multi_objective:
            # TODO add possibility of using regularization in MO alg
            self.regularization_type = RegularizationTypesEnum.none
