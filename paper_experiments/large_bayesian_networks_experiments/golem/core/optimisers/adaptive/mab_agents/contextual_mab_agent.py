import random
from functools import partial
from typing import Union, Sequence, Optional, List, Callable

import numpy as np
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
from scipy.special import softmax

from golem.core.dag.graph import Graph
from golem.core.dag.graph_node import GraphNode
from golem.core.optimisers.adaptive.context_agents import ContextAgentsRepository, ContextAgentTypeEnum
from golem.core.optimisers.adaptive.mab_agents.mab_agent import MultiArmedBanditAgent
from golem.core.optimisers.adaptive.operator_agent import ActType, ObsType, ExperienceBuffer


class ContextualMultiArmedBanditAgent(MultiArmedBanditAgent):
    """ Contextual Multi-Armed bandit. Observations can be encoded with simple context agent without
    using NN to guarantee convergence.

    :param actions: types of mutations
    :param context_agent_type: function to convert observation to its embedding. Can be specified as
    ContextAgentTypeEnum or as Callable function.
    :param available_operations: available operations
    :param n_jobs: n_jobs
    :param enable_logging: bool logging flag
    """

    def __init__(self, actions: Sequence[ActType],
                 context_agent_type: Union[ContextAgentTypeEnum, Callable],
                 available_operations: List[str],
                 n_jobs: int = 1,
                 enable_logging: bool = True,
                 decaying_factor: float = 1.0):
        super().__init__(actions=actions, n_jobs=n_jobs, enable_logging=enable_logging,
                         decaying_factor=decaying_factor, is_initial_fit=False)
        self._agent = MAB(arms=self._indices,
                          learning_policy=LearningPolicy.UCB1(alpha=1.25),
                          neighborhood_policy=NeighborhoodPolicy.Clusters(),
                          n_jobs=n_jobs)
        self._context_agent = context_agent_type if isinstance(context_agent_type, Callable) else \
            partial(ContextAgentsRepository.agent_class_by_id(context_agent_type),
                    available_operations=available_operations)
        self._is_fitted = False

    def _initial_fit(self, obs: ObsType):
        """ Initial fit for Contextual Multi-Armed Bandit.
        At this step, all hands are assigned the same weights with the very first context
        that is fed to the bandit. """
        # initial fit for mab
        n = len(self._indices)
        uniform_rewards = [1. / n] * n
        contexts = self.get_context(obs=obs)
        self._agent.fit(decisions=self._indices, rewards=uniform_rewards, contexts=np.tile(contexts, (n, 1)))
        self._is_fitted = True

    def choose_action(self, obs: ObsType) -> ActType:
        if not self._is_fitted:
            self._initial_fit(obs=obs)
        contexts = self.get_context(obs=obs)
        arm = self._agent.predict(contexts=contexts.reshape(1, -1))
        action = self.actions[arm]
        return action

    def get_action_values(self, obs: Optional[ObsType] = None) -> Sequence[float]:
        if not self._is_fitted:
            self._initial_fit(obs=obs)
        contexts = self.get_context(obs)
        prob_dict = self._agent.predict_expectations(contexts=contexts.reshape(1, -1))
        prob_list = [prob_dict[i] for i in range(len(prob_dict))]
        return prob_list

    def get_action_probs(self, obs: Optional[ObsType] = None) -> Sequence[float]:
        return softmax(self.get_action_values(obs=obs))

    def choose_nodes(self, graph: Graph, num_nodes: int = 1) -> Union[GraphNode, Sequence[GraphNode]]:
        subject_nodes = random.sample(graph.nodes, k=num_nodes)
        return subject_nodes[0] if num_nodes == 1 else subject_nodes

    def partial_fit(self, experience: ExperienceBuffer):
        """Continues learning of underlying agent with new experience."""
        obs, arms, processed_rewards = self._get_experience(experience)
        contexts = self.get_context(obs=obs)
        self._agent.partial_fit(decisions=arms, rewards=processed_rewards, contexts=contexts)

    def _get_experience(self, experience: ExperienceBuffer):
        """ Get experience from ExperienceBuffer, process rewards and log. """
        obs, actions, rewards = experience.retrieve_experience()
        arms = [self._arm_by_action[action.__name__] for action in actions]
        # there is no need to process rewards as in MAB, since this processing unifies rewards for all contexts
        self._dbg_log(obs, actions, rewards)
        return obs, arms, rewards

    def get_context(self, obs: Union[List[ObsType], ObsType]) -> np.array:
        """ Returns contexts based on specified context agent. """
        if not isinstance(obs, list):
            obs = [obs]
        contexts = []
        for ob in obs:
            if isinstance(ob, list) or isinstance(ob, np.ndarray):
                # to unify type to list
                contexts.append(np.array(ob).flatten())
            else:
                context = np.array(self._context_agent(ob))
                # some external context agents can wrap context in an additional array
                contexts.append(context.flatten())
        return np.array(contexts)
