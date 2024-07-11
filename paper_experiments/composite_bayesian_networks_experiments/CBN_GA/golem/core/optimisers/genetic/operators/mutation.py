from copy import deepcopy
from random import random
from typing import Callable, Union, Tuple, TYPE_CHECKING, Mapping, Hashable, Optional

import numpy as np

from golem.core.dag.graph import Graph
from golem.core.optimisers.adaptive.mab_agents.contextual_mab_agent import ContextualMultiArmedBanditAgent
from golem.core.optimisers.adaptive.mab_agents.neural_contextual_mab_agent import NeuralContextualMultiArmedBanditAgent
from golem.core.optimisers.adaptive.mab_agents.mab_agent import MultiArmedBanditAgent
from golem.core.optimisers.adaptive.operator_agent import \
    OperatorAgent, RandomAgent, ExperienceBuffer, MutationAgentTypeEnum
from golem.core.optimisers.genetic.operators.base_mutations import \
    base_mutations_repo, MutationTypesEnum
from golem.core.optimisers.genetic.operators.operator import PopulationT, Operator
from golem.core.optimisers.graph import OptGraph
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
                available_operations=graph_gen_params.node_factory.available_nodes,
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

        final_population, mutations_applied, application_attempts = tuple(zip(*map(self._mutation, population)))

        # [print(attempt, ind.graph != init_ind.graph, not attempt or ind.graph != init_ind.graph) for ind, init_ind, attempt in zip(final_population, population, application_attempts)
        #                     ]
        
        # drop individuals to which mutations could not be applied
        final_population = [ind for ind, init_ind, attempt in zip(final_population, population, application_attempts)
                            if not(attempt and ind.graph == init_ind.graph and {node:node.content['parent_model'] for node in ind.graph.nodes} == {node:node.content['parent_model'] for node in init_ind.graph.nodes}) # mine
                            # if attempt 
                            
                            # if not attempt or ind.graph != init_ind.graph
                            # if not(attempt and ind.graph == init_ind.graph) # так в новой версии 
                            ]

        if len(population) == 1: # mine
            return final_population[0] if final_population else final_population

        return final_population

    def _mutation(self, individual: Individual) -> Tuple[Individual, Optional[MutationIdType], bool]:
        """ Function applies mutation operator to graph """
        application_attempt = False
        mutation_applied = None
        for _ in range(self.parameters.max_num_of_operator_attempts):
            new_graph = deepcopy(individual.graph)

            new_graph, mutation_applied = self._apply_mutations(new_graph)
            if mutation_applied is None:
                continue
            application_attempt = True
            is_correct_graph = self.graph_generation_params.verifier(new_graph)
            if is_correct_graph:
                parent_operator = ParentOperator(type_='mutation',
                                                 operators=mutation_applied,
                                                 parent_individuals=individual)
                individual = Individual(new_graph, parent_operator,
                                        metadata=self.requirements.static_individual_metadata)
                break
            else:
                # Collect invalid actions
                self.agent_experience.collect_experience(individual.graph, mutation_applied, reward=-1.0)
        else:
            self.log.debug('Number of mutation attempts exceeded. '
                           'Please check optimization parameters for correctness.')
        return individual, mutation_applied, application_attempt

    def _sample_num_of_mutations(self) -> int:
        # most of the time returns 1 or rarely several mutations
        if self.parameters.variable_mutation_num:
            num_mut = max(int(round(np.random.lognormal(0, sigma=0.5))), 1)
        else:
            num_mut = 1
        return num_mut

    def _apply_mutations(self, new_graph: OptGraph) -> Tuple[OptGraph, Optional[MutationIdType]]:
        """Apply mutation 1 or few times iteratively"""
        mutation_type = self._operator_agent.choose_action(new_graph)
        mutation_applied = None
        for _ in range(self._sample_num_of_mutations()):
            new_graph, applied = self._adapt_and_apply_mutation(new_graph, mutation_type)
            if applied:
                mutation_applied = mutation_type
                is_custom_mutation = isinstance(mutation_type, Callable)
                if is_custom_mutation:  # custom mutation occurs once
                    break
        return new_graph, mutation_applied

    def _adapt_and_apply_mutation(self, new_graph: OptGraph, mutation_type) -> Tuple[OptGraph, bool]:
        applied = self._will_mutation_be_applied(mutation_type)
        if applied:
            # get the mutation function and adapt it
            mutation_func = self._get_mutation_func(mutation_type)
            new_graph = mutation_func(new_graph, requirements=self.requirements,
                                      graph_gen_params=self.graph_generation_params,
                                      parameters=self.parameters)
        return new_graph, applied

    def _will_mutation_be_applied(self, mutation_type: Union[MutationTypesEnum, Callable]) -> bool:
        return random() <= self.parameters.mutation_prob and mutation_type is not MutationTypesEnum.none

    def _get_mutation_func(self, mutation_type: Union[MutationTypesEnum, Callable]) -> Callable:
        if isinstance(mutation_type, Callable):
            mutation_func = mutation_type
        else:
            mutation_func = self._mutations_repo[mutation_type]
        adapted_mutation_func = self.graph_generation_params.adapter.adapt_func(mutation_func)
        return adapted_mutation_func
