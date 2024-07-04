from typing import Union, Optional, Sequence

from golem.core.dag.graph import Graph
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.mutation import Mutation
from golem.core.optimisers.genetic.operators.operator import EvaluationOperator, PopulationT
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.populational_optimizer import PopulationalOptimizer
from golem.core.optimisers.random.random_search import RandomSearchOptimizer
from golem.utilities.data_structures import ensure_wrapped_in_sequence


class PopulationalRandomMutationOptimizer(PopulationalOptimizer):
    """
    Populational random search-based graph models optimizer.
    Implemented as a baseline to compare with evolutionary algorithm.
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
        self.initial_individuals = [Individual(graph, metadata=requirements.static_individual_metadata)
                                    for graph in self.initial_graphs]
        self.mutation = Mutation(self.graph_optimizer_params, self.requirements, self.graph_generation_params)

    def _evolve_population(self, evaluator: EvaluationOperator) -> PopulationT:
        new_population = ensure_wrapped_in_sequence(self.mutation(self.population))
        new_population = evaluator(new_population)
        return new_population

    def _initial_population(self, evaluator: EvaluationOperator):
        self._update_population(evaluator(self.initial_individuals), 'initial_assumptions')
        pop_size = self.graph_optimizer_params.pop_size

        if len(self.initial_individuals) < pop_size:
            self.initial_individuals = self._extend_population(self.initial_individuals, pop_size)
            # Adding of extended population to history
            self._update_population(evaluator(self.initial_individuals), 'extended_initial_assumptions')


class RandomMutationOptimizer(RandomSearchOptimizer):
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
