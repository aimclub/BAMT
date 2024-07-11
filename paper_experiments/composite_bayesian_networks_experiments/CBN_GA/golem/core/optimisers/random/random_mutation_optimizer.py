from typing import Union, Optional, Sequence

from golem.core.dag.graph import Graph
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.mutation import Mutation
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.random.random_search import RandomSearchOptimizer


class RandomMutationSearchOptimizer(RandomSearchOptimizer):
    """
    Random search-based graph models optimizer
    """

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Union[Graph, Sequence[Graph]],
                 requirements: Optional[GraphRequirements] = None,
                 graph_generation_params: Optional[GraphGenerationParams] = None,
                 graph_optimizer_params: Optional[GPAlgorithmParameters] = None):
        requirements = requirements or GraphRequirements()
        graph_optimizer_params = graph_optimizer_params or GPAlgorithmParameters()
        super().__init__(objective, initial_graphs, requirements, graph_generation_params, graph_optimizer_params)
        self.mutation = Mutation(self.graph_optimizer_params, self.requirements, self.graph_generation_params)

    def _generate_new_individual(self) -> Individual:
        return self.mutation(self.best_individual)
