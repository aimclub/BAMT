from copy import deepcopy
from random import random
from typing import Callable, Union, Tuple, TYPE_CHECKING, Mapping, Hashable, Optional

import numpy as np

from golem.core.dag.graph import Graph
from golem.core.optimisers.adaptive.mab_agents.contextual_mab_agent import ContextualMultiArmedBanditAgent
from golem.core.optimisers.adaptive.mab_agents.mab_agent import MultiArmedBanditAgent
from golem.core.optimisers.adaptive.mab_agents.neural_contextual_mab_agent import NeuralContextualMultiArmedBanditAgent
from golem.core.optimisers.adaptive.operator_agent import \
    OperatorAgent, RandomAgent, MutationAgentTypeEnum
from golem.core.optimisers.adaptive.experience_buffer import ExperienceBuffer
from golem.core.optimisers.genetic.operators.base_mutations import \
    base_mutations_repo, MutationTypesEnum
from golem.core.optimisers.genetic.operators.operator import PopulationT, Operator
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator
from golem.core.optimisers.optimization_parameters import GraphRequirements, OptimizationParameters
from golem.core.optimisers.optimizer import GraphGenerationParams, AlgorithmParameters

if TYPE_CHECKING:
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters

MutationFunc = Callable[[Graph, GraphRequirements, GraphGenerationParams, AlgorithmParameters], Graph]
MutationIdType = Hashable
MutationRepo = Mapping[MutationIdType, MutationFunc]


class Mutation(Operator):
    def __init__(self,
                 parameters: 'GPAlgorithmParameters',
                 requirements: GraphRequirements,
                 graph_gen_params: GraphGenerationParams,
                 mutations_repo: Optional[MutationRepo] = None,
                 ):
        super().__init__(parameters, requirements)
        self.graph_generation_params = graph_gen_params
        self.parameters = parameters
        self._mutations_repo = mutations_repo or base_mutations_repo
        self._operator_agent = self._init_operator_agent(graph_gen_params, parameters, requirements)
        self.agent_experience = ExperienceBuffer(window_size=parameters.window_size)

    @staticmethod
    def _init_operator_agent(graph_gen_params: GraphGenerationParams,
                             parameters: 'GPAlgorithmParameters',
                             requirements: OptimizationParameters):
        kind = parameters.adaptive_mutation_type
        if kind == MutationAgentTypeEnum.default or kind == MutationAgentTypeEnum.random:
            agent = RandomAgent(actions=parameters.mutation_types)
        elif kind == MutationAgentTypeEnum.bandit:
            agent = MultiArmedBanditAgent(actions=parameters.mutation_types,
                                          n_jobs=requirements.n_jobs,
                                          path_to_save=requirements.agent_dir,
                                          decaying_factor=parameters.decaying_factor)
        elif kind == MutationAgentTypeEnum.contextual_bandit:
            agent = ContextualMultiArmedBanditAgent(
                actions=parameters.mutation_types,
                context_agent_type=parameters.context_agent_type,
                available_operations=graph_gen_params.node_factory.get_all_available_operations(),
                n_jobs=requirements.n_jobs,
                decaying_factor=parameters.decaying_factor)
        elif kind == MutationAgentTypeEnum.neural_bandit:
            agent = NeuralContextualMultiArmedBanditAgent(
                actions=parameters.mutation_types,
                context_agent_type=parameters.context_agent_type,
                available_operations=graph_gen_params.node_factory.get_all_available_operations(),
                n_jobs=requirements.n_jobs)
        # if agent was specified pretrained (with instance)
        elif isinstance(parameters.adaptive_mutation_type, OperatorAgent):
            agent = kind
        else:
            raise TypeError(f'Unknown parameter {kind}')
        return agent

    @property
    def agent(self) -> OperatorAgent:
        return self._operator_agent

    def __call__(self, population: Union[Individual, PopulationT]) -> Union[Individual, PopulationT]:
        if isinstance(population, Individual):
            population = [population]

        final_population, application_attempts = tuple(zip(*map(self._mutation, population)))

        # drop individuals to which mutations could not be applied
        final_population = [ind for ind, init_ind, attempt in zip(final_population, population, application_attempts)
                            if not(attempt and ind.graph == init_ind.graph)]

        if len(population) == 1:
            return final_population[0] if final_population else final_population

        return final_population

    def _mutation(self, individual: Individual) -> Tuple[Individual, bool]:
        """ Function applies mutation operator to graph """
        mutation_type = self._operator_agent.choose_action(individual.graph)
        is_applied = self._will_mutation_be_applied(mutation_type)
        if is_applied:
            for _ in range(self.parameters.max_num_of_operator_attempts):
                new_graph = deepcopy(individual.graph)

                new_graph = self._apply_mutations(new_graph, mutation_type)
                is_correct_graph = self.graph_generation_params.verifier(new_graph)
                if is_correct_graph:
                    # str for custom mutations serialisation
                    parent_operator = ParentOperator(type_='mutation',
                                                     operators=mutation_type.__name__,
                                                     parent_individuals=individual)
                    individual = Individual(new_graph, parent_operator,
                                            metadata=self.requirements.static_individual_metadata)
                    break
            else:
                # Collect invalid actions
                self.agent_experience.collect_experience(individual, mutation_type, reward=-1.0)

                self.log.debug(f'Number of attempts for {mutation_type} mutation application exceeded. '
                               'Please check optimization parameters for correctness.')
        return individual, is_applied

    def _sample_num_of_mutations(self, mutation_type: Union[MutationTypesEnum, Callable]) -> int:
        # most of the time returns 1 or rarely several mutations
        is_custom_mutation = isinstance(mutation_type, Callable)
        if self.parameters.variable_mutation_num and not is_custom_mutation:
            num_mut = max(int(round(np.random.lognormal(0, sigma=0.5))), 1)
        else:
            num_mut = 1
        return num_mut

    def _apply_mutations(self, new_graph: Graph, mutation_type: Union[MutationTypesEnum, Callable]) -> Graph:
        """Apply mutation 1 or few times iteratively"""
        for _ in range(self._sample_num_of_mutations(mutation_type)):
            mutation_func = self._get_mutation_func(mutation_type)
            new_graph = mutation_func(new_graph, requirements=self.requirements,
                                      graph_gen_params=self.graph_generation_params,
                                      parameters=self.parameters)
        return new_graph

    def _will_mutation_be_applied(self, mutation_type: Union[MutationTypesEnum, Callable]) -> bool:
        return random() <= self.parameters.mutation_prob and mutation_type is not MutationTypesEnum.none

    def _get_mutation_func(self, mutation_type: Union[MutationTypesEnum, Callable]) -> Callable:
        if isinstance(mutation_type, Callable):
            mutation_func = mutation_type
        else:
            mutation_func = self._mutations_repo[mutation_type]
        adapted_mutation_func = self.graph_generation_params.adapter.adapt_func(mutation_func)
        return adapted_mutation_func
