from functools import partial
from typing import Callable, Optional, Sequence, Union, Iterable

from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.constants import MAX_GRAPH_GEN_ATTEMPTS
from golem.core.dag.graph import Graph
from golem.core.log import default_log
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.optimizer import GraphGenerationParams

GenerationFunction = Callable[[], Graph]
InitialGraphsGenerator = Callable[[], Sequence[OptGraph]]


class InitialPopulationGenerator(InitialGraphsGenerator):
    """Generates initial population using three approaches.
    One is with initial graphs.
    Another is with initial graphs generation function which generates a graph
    that will be added to initial population.
    The third way is random graphs generation according to GraphGenerationParameters and OptimizationParameters.
    The last approach is applied when neither initial graphs nor initial graphs generation function were provided."""

    def __init__(self,
                 population_size: int,
                 generation_params: GraphGenerationParams,
                 requirements: GraphRequirements):
        self.pop_size = population_size
        self.requirements = requirements
        self.graph_generation_params = generation_params
        self.generation_function: Optional[GenerationFunction] = None
        self.initial_graphs: Optional[Sequence[Graph]] = None
        self.log = default_log(self)

    def __call__(self) -> Sequence[OptGraph]:
        pop_size = self.pop_size
        adapter = self.graph_generation_params.adapter

        if self.initial_graphs:
            if len(self.initial_graphs) > pop_size:
                self.initial_graphs = self.initial_graphs[:pop_size]
            return adapter.adapt(self.initial_graphs)

        if not self.generation_function:
            self.generation_function = partial(self.graph_generation_params.random_graph_factory, self.requirements)

        population = []
        for iter_num in range(MAX_GRAPH_GEN_ATTEMPTS):
            if len(population) == pop_size:
                break
            new_graph = self.generation_function()
            if new_graph not in population and self.graph_generation_params.verifier(new_graph):
                population.append(new_graph)
        else:
            self.log.warning(f'Exceeded max number of attempts for generating initial graphs, stopping.'
                             f'Generated {len(population)} instead of {pop_size} graphs.')
        return population

    def with_initial_graphs(self, initial_graphs: Union[Graph, Sequence[Graph]]):
        """Use initial graphs as initial population."""
        if isinstance(initial_graphs, Graph):
            self.initial_graphs = [initial_graphs]
        elif isinstance(initial_graphs, Iterable):
            self.initial_graphs = list(initial_graphs)
        else:
            raise ValueError(f'Incorrect type of initial_assumption: '
                             f'Sequence[Graph] or Graph needed, but has {type(initial_graphs)}')
        return self

    def with_custom_generation_function(self, generation_func: GenerationFunction):
        """Use custom graph generation function to create initial population."""
        self.generation_function = generation_func
        return self
