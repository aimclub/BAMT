import logging
from typing import Optional

from golem.api.api_utils.api_params import ApiParams
from golem.core.constants import DEFAULT_API_TIMEOUT_MINUTES
from golem.core.log import Log, default_log
from golem.utilities.utilities import set_random_seed


class GOLEM:
    """
    Main class for GOLEM API.

    Args:
        :param timeout: timeout for optimization.
        :param seed: value for a fixed random seed.
        :param logging_level: logging levels are the same as in `logging <https://docs.python.org/3/library/logging.html>`_.

            .. details:: Possible options:

                - ``50`` -> critical
                - ``40`` -> error
                - ``30`` -> warning
                - ``20`` -> info
                - ``10`` -> debug
                - ``0`` -> nonset
        :param n_jobs: num of ``n_jobs`` for parallelization (set to ``-1`` to use all cpu's). Defaults to ``-1``.
        :param graph_requirements_class: class to specify custom graph requirements.
        Must be inherited from GraphRequirements class.

        :param crossover_prob: crossover probability (chance that two individuals will be mated).

        ``GPAlgorithmParameters`` parameters
        :param mutation_prob: mutation probability (chance that an individual will be mutated).
        :param variable_mutation_num: flag to apply mutation one or few times for individual in each iteration.
        :param max_num_of_operator_attempts: max number of unsuccessful evo operator attempts before continuing.
        :param mutation_strength: strength of mutation in tree (using in certain mutation types)
        :param min_pop_size_with_elitism: minimal population size with which elitism is applicable
        :param required_valid_ratio: ratio of valid individuals on next population to continue optimization.

        Used in `ReproductionController` to compensate for invalid individuals. See the class for details.

        :param adaptive_mutation_type: enables adaptive Mutation agent.
        :param context_agent_type: enables graph encoding for Mutation agent.

        Adaptive mutation agent uses specified algorithm. 'random' type is the default non-adaptive version.
        Requires crossover_types to be CrossoverTypesEnum.none for correct adaptive learning,
        so that fitness changes depend only on agent's actions (chosen mutations).
        ``MutationAgentTypeEnum.bandit`` uses Multi-Armed Bandit (MAB) learning algorithm.
        ``MutationAgentTypeEnum.contextual_bandit`` uses contextual MAB learning algorithm.
        ``MutationAgentTypeEnum.neural_bandit`` uses contextual MAB learning algorithm with Deep Neural encoding.

        Parameter `context_agent_type` specifies implementation of graph/node encoder for adaptive
        mutation agent. It is relevant for contextual and neural bandits.

        :param decaying_factor: decaying factor for Multi-Armed Bandits for managing the profit from operators
        The smaller the value of decaying_factor, the larger the influence for the best operator.
        :param window_size: the size of sliding window for Multi-Armed Bandits to decrease variance.
        The window size is measured by the number of individuals to consider.


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

        In the `steady_state` individuals from previous populations are mixed with the ones from new population.
        UUIDs of individuals do not repeat within one population.

        The `parameter_free` scheme is same as `steady_state` for now.

        ``GraphGenerationParams`` parameters
        :param adapter: instance of domain graph adapter for adaptation
         between domain and optimization graphs
        :param rules_for_constraint: collection of constraints for graph verification
        :param advisor: instance providing task and context-specific advices for graph changes
        :param node_factory: instance for generating new nodes in the process of graph search
        :param remote_evaluator: instance of delegate evaluator for evaluation of graphs

        ``GraphRequirements`` parameters
        :param start_depth: start value of adaptive tree depth
        :param max_depth: max depth of the resulting graph
        :param min_arity: min number of parents for node
        :param max_arity: max number of parents for node

        Also, custom domain specific parameters can be specified here. These parameters can be then used in
        ``DynamicGraphRequirements`` as fields.
    """
    def __init__(self,
                 timeout: Optional[float] = DEFAULT_API_TIMEOUT_MINUTES,
                 seed: Optional[int] = None,
                 logging_level: int = logging.INFO,
                 n_jobs: int = -1,
                 **all_parameters):
        set_random_seed(seed)
        self.log = self._init_logger(logging_level)

        self.api_params = ApiParams(input_params=all_parameters,
                                    n_jobs=n_jobs,
                                    timeout=timeout)
        self.gp_algorithm_parameters = self.api_params.get_gp_algorithm_parameters()
        self.graph_generation_parameters = self.api_params.get_graph_generation_parameters()
        self.graph_requirements = self.api_params.get_graph_requirements()

    def optimise(self, **custom_optimiser_parameters):
        """ Method to start optimisation process.
        `custom_optimiser_parameters` parameters can be specified additionally to use it directly in optimiser.
        """
        common_params = self.api_params.get_actual_common_params()
        optimizer_cls = common_params['optimizer']
        objective = common_params['objective']
        initial_graphs = common_params['initial_graphs']

        self.optimiser = optimizer_cls(objective,
                                       initial_graphs,
                                       self.graph_requirements,
                                       self.graph_generation_parameters,
                                       self.gp_algorithm_parameters,
                                       **custom_optimiser_parameters)

        found_graphs = self.optimiser.optimise(objective)
        return found_graphs

    @staticmethod
    def _init_logger(logging_level: int):
        # reset logging level for Singleton
        Log().reset_logging_level(logging_level)
        return default_log(prefix='GOLEM logger')
