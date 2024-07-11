import random
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from itertools import chain
from typing import Union, Sequence, Hashable, Tuple, Optional, List

import numpy as np

from golem.core.dag.graph import Graph
from golem.core.dag.graph_node import GraphNode
from golem.core.log import default_log
from golem.core.optimisers.opt_history_objects.individual import Individual

ObsType = Graph
ActType = Hashable


class MutationAgentTypeEnum(Enum):
    default = 'default'
    random = 'random'
    bandit = 'bandit'
    contextual_bandit = 'contextual_bandit'
    neural_bandit = 'neural_bandit'


class ExperienceBuffer:
    """
    Buffer for learning experience of ``OperatorAgent``.
    Keeps (State, Action, Reward) lists until retrieval.
    Can be used with window_size for actualizing experience.
    """

    def __init__(self, window_size: Optional[int] = None):
        self.window_size = window_size
        self._reset_main_storages()
        self.reset()

    def reset(self):
        self._current_observations = []
        self._current_actions = []
        self._current_rewards = []
        self._prev_pop = set()
        self._next_pop = set()

        # if window size was not specified than there is no need to store these values for reuse
        if self.window_size is None:
            self._reset_main_storages()

    def _reset_main_storages(self):
        self._observations = deque(maxlen=self.window_size)
        self._actions = deque(maxlen=self.window_size)
        self._rewards = deque(maxlen=self.window_size)

    def collect_results(self, results: Sequence[Individual]):
        for ind in results:
            self.collect_result(ind)
        self._observations += self._current_observations
        self._actions += self._current_actions
        self._rewards += self._current_rewards

    def collect_result(self, result: Individual):
        if result.uid in self._prev_pop:
            return
        if not result.parent_operator or result.parent_operator.type_ != 'mutation':
            return
        self._next_pop.add(result.uid)
        obs = result.graph
        action = result.parent_operator.operators[0]
        prev_fitness = result.parent_operator.parent_individuals[0].fitness.value
        # we're minimising the fitness, that's why less is better
        # reward is defined as fitness improvement rate (FIR) to stabilize the algorithm
        reward = (prev_fitness - result.fitness.value) / abs(prev_fitness) \
            if prev_fitness is not None and prev_fitness != 0 else 0.
        self.collect_experience(obs, action, reward)

    def collect_experience(self, obs: ObsType, action: ActType, reward: float):
        self._current_observations.append(obs)
        self._current_actions.append(action)
        self._current_rewards.append(reward)

    def retrieve_experience(self) -> Tuple[List[ObsType], List[ActType], List[float]]:
        """Get all collected experience and clear the experience buffer."""
        observations, actions, rewards = self._observations, self._actions, self._rewards
        next_pop = self._next_pop
        self.reset()
        self._prev_pop = next_pop
        return list(observations), \
            list(actions), \
            list(rewards)


class OperatorAgent(ABC):
    def __init__(self, enable_logging: bool = True):
        self._enable_logging = enable_logging
        self._log = default_log(self)

    @abstractmethod
    def partial_fit(self, experience: ExperienceBuffer):
        raise NotImplementedError()

    @abstractmethod
    def choose_action(self, obs: Optional[ObsType]) -> ActType:
        raise NotImplementedError()

    @abstractmethod
    def choose_nodes(self, graph: Graph, num_nodes: int = 1) -> Union[GraphNode, Sequence[GraphNode]]:
        raise NotImplementedError()

    @abstractmethod
    def get_action_probs(self, obs: Optional[ObsType]) -> Sequence[float]:
        raise NotImplementedError()

    @abstractmethod
    def get_action_values(self, obs: Optional[ObsType]) -> Sequence[float]:
        raise NotImplementedError()

    def _dbg_log(self, obs, actions, rewards):
        if self._enable_logging:
            prec = 4
            rr = np.array(rewards).round(prec)
            nonzero = rr[rr.nonzero()]
            msg = f'len={len(rr)} nonzero={len(nonzero)} '
            if len(nonzero) > 0:
                msg += (f'avg={nonzero.mean()} std={nonzero.std()} '
                        f'min={nonzero.min()} max={nonzero.max()} ')

            self._log.info(msg)
            self._log.info(f'actions/rewards: {list(zip(actions, rr))}')

            action_values = list(map(self.get_action_values, obs))
            action_probs = list(map(self.get_action_probs, obs))
            action_values = np.round(np.mean(action_values, axis=0), prec)
            action_probs = np.round(np.mean(action_probs, axis=0), prec)

            self._log.info(f'exp={action_values} '
                           f'probs={action_probs}')


class RandomAgent(OperatorAgent):
    def __init__(self,
                 actions: Sequence[ActType],
                 probs: Optional[Sequence[float]] = None,
                 enable_logging: bool = True):
        self.actions = list(actions)
        self._probs = probs or [1. / len(actions)] * len(actions)
        super().__init__(enable_logging)

    def choose_action(self, obs: ObsType) -> ActType:
        action = np.random.choice(self.actions, p=self.get_action_probs(obs))
        return action

    def choose_nodes(self, graph: Graph, num_nodes: int = 1) -> Union[GraphNode, Sequence[GraphNode]]:
        subject_nodes = random.sample(graph.nodes, k=num_nodes)
        return subject_nodes[0] if num_nodes == 1 else subject_nodes

    def partial_fit(self, experience: ExperienceBuffer):
        obs, actions, rewards = experience.retrieve_experience()
        self._dbg_log(obs, actions, rewards)

    def get_action_probs(self, obs: Optional[ObsType] = None) -> Sequence[float]:
        return self._probs

    def get_action_values(self, obs: Optional[ObsType] = None) -> Sequence[float]:
        return self._probs
